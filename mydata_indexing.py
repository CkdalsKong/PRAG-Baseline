import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, PROMPT_LLM_FILTERING, PROMPT_LLM_SUMMARIZING, INDEXING_REPORT_FILE, THRESHOLD
# from HippoRAG import HippoRAG

import random

class MyDataIndexing(MyDataUtils):
    def __init__(self, utils):
        super().__init__(
            mode=utils.mode,
            method=utils.method,
            device=utils.device,
            use_multi_gpu=utils.use_multi_gpu,
            chunk_mode=utils.chunk_mode,
            output_dir=utils.output_dir,
            persona_task_file=utils.persona_task_file,
            emb_model_name=utils.emb_model_name
        )
        self.utils = utils
        
        # 임베딩 파일 경로 설정
        self.embeddings_file = os.path.join(self.output_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        self.index_file = os.path.join(self.output_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")
    
    def run_indexing(self, persona_index):
        if persona_index == -1:
            print(f"\n=== Starting indexing with method {self.method} ===")
        else:
            print(f"\n=== Starting indexing for persona {persona_index} with method {self.method} ===")
        
        # 출력 디렉토리 설정
        if self.method in ["standard", "random", "hipporag"]:
            method_dir = os.path.join(self.output_dir, f"{self.method}")
        else:
            method_dir = os.path.join(self.output_dir, f"{self.method}/{persona_index}")
        os.makedirs(method_dir, exist_ok=True)
        print(f"Output directory: {method_dir}")
        
        # 데이터 로드
        if self.method in ["naive_p", "cosine_only"]:
            persona = self.load_persona_data(persona_index)
            print(f"Loaded persona data for index {persona_index}")
        
        self.load_models()

        # chunks와 embeddings 로드
        print("Loading chunks and embeddings...")
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            chunks = [json.loads(line)["text"] for line in f]
        chunk_embeddings = np.load(self.embedding_file)
        print(f"Loaded {len(chunks)} chunks and their embeddings")
        
        # Preference embeddings 생성 또는 로드 (naive_p, cosine_only 방식만)
        if self.method in ["naive_p", "cosine_only"]:
            print("Loading or generating preference embeddings...")
            preference_list = [block["preference"] for block in persona["preference_blocks"]]
            preference_emb_file = os.path.join(method_dir, "preference_embeddings.npy")
            
            if os.path.exists(preference_emb_file):
                print("Loading existing preference embeddings...")
                preference_embeddings = np.load(preference_emb_file)
            else:
                print("Generating new preference embeddings...")
                preference_embeddings = self.embed_texts(preference_list)
                preference_embeddings = preference_embeddings / np.linalg.norm(preference_embeddings, axis=1, keepdims=True)
                np.save(preference_emb_file, preference_embeddings)
                print(f"Saved preference embeddings to {preference_emb_file}")
            
            chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            print(f"Using {len(preference_list)} preference embeddings")
        
        start_total = time.time()

        # Cosine similarity 기반 필터링 (naive_p, cosine_only 방식)
        if self.method in ["naive_p", "cosine_only"]:
            print("\nStarting cosine similarity filtering...")
            kept_save, kept_chunks, filtered_save = [], [], []
            keep_indices = []
            
            # 배치 단위로 처리
            batch_size = self.batch_size
            for i in tqdm(range(0, len(chunk_embeddings_norm), batch_size), desc=f"Filtering persona {persona_index}"):
                batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                
                # 배치 단위로 유사도 계산
                sims = np.dot(preference_embeddings, batch_embeddings.T)
                mask = np.any(sims > THRESHOLD, axis=0)
                
                # 결과 분류
                for j, (chunk, is_kept) in enumerate(zip(batch_chunks, mask)):
                    if is_kept:
                        kept_chunks.append(chunk)
                        kept_save.append({"chunk": chunk})
                    else:
                        filtered_save.append({"chunk": chunk})
            
            filter_time = time.time() - start_total
            print(f"Cosine filtering completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
            
        elif self.method == "random":
            print("\nStarting random filtering...")
            # 전체 청크에서 10% 랜덤 샘플링
            num_samples = int(len(chunks) * 0.1)
            
            # numpy를 사용한 빠른 랜덤 샘플링
            indices = np.random.choice(len(chunks), size=num_samples, replace=False)
            kept_chunks = [chunks[i] for i in indices]
            kept_save = [{"chunk": chunk} for chunk in kept_chunks]
            
            # filtered_save는 나중에 필요할 때 생성
            filtered_save = []
            
            filter_time = time.time() - start_total
            print(f"Random sampling completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
        # elif self.method == "hipporag":
        #     print("\nStarting HippoRAG indexing...")
        #     hipporag = HippoRAG(device=self.device, use_multi_gpu=self.use_multi_gpu)
        #     hipporag.load_model()
        #     filter_time = hipporag.index(chunks, method_dir)
        #     kept_chunks = chunks  # HippoRAG는 모든 청크를 사용
        #     kept_save = [{"chunk": chunk} for chunk in chunks]
        #     filtered_save = []
        else:  # standard 방식
            filter_time = 0
            kept_chunks = chunks
            print("Skipping cosine filtering for standard method")

        # LLM 필터링 (naive_p 방식)
        if self.method in ["naive_p"]:
            print("\nStarting LLM filtering...")
            # Prompt 로드
            filtering_prompt = self.load_prompt_template(PROMPT_LLM_FILTERING)
            summarizing_prompt = self.load_prompt_template(PROMPT_LLM_SUMMARIZING)
            
            # preference_list를 문자열로 변환
            preference_text = "\n".join([f"- {p}" for p in preference_list])
            start_llm = time.time()
            results = []
            
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.process_chunk, kept_chunks[idx], preference_text, filtering_prompt): idx for idx in range(len(kept_chunks))}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}"):
                    result = future.result()
                    if result:
                        results.append(result)

            result_info_file = os.path.join(method_dir, "result_info.jsonl")
            self.save_json(result_info_file, results)
            print(f"✅ Result info saved to {result_info_file}")

            # 결과 분류
            filtered, summarized, kept = [], [], []
            for item in results:
                if item["decision"] == "Filter":
                    filtered.append({"chunk": item["chunk"]})
                elif item["decision"] == "Summarize":
                    summarized.append({
                        "chunk": item["chunk"],
                        "reason": item["reason"]
                    })
                elif item["decision"] == "Keep As-Is":
                    kept.append({"chunk": item["chunk"]})
            
            llm_time = time.time() - start_llm
            print(f"LLM filtering completed. Filtered: {len(filtered)}, Summarized: {len(summarized)}, Kept: {len(kept)}")
            
            # Summarization (naive_p 방식만)
            if self.method == "naive_p":
                summarized_file = os.path.join(method_dir, "summarized.jsonl")
                print("\nStarting summarization...")
                start_summary = time.time()
                summarized_final = []
                with ThreadPoolExecutor() as executor:  
                    futures = [executor.submit(self.summarize_single, entry, summarizing_prompt) for entry in summarized]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing chunks"):
                        try:
                            result = future.result()
                            summarized_final.append(result)
                        except Exception as e:
                            print(f"Summarization failed: {e}")
                summary_time = time.time() - start_summary
                self.save_jsonl(summarized_file, summarized_final)
                print(f"Summarization completed. Summarized {len(summarized_final)} chunks")
                print(f"✅ Summary info saved to {summarized_file}")
                
                # 최종 청크 병합 (요약된 텍스트만 사용)
                merged_chunks = [item["summarized"] for item in summarized_final] + [item["chunk"] for item in kept]
            else:
                merged_chunks = [item["chunk"] for item in kept]
                summary_time = 0
        else:
            merged_chunks = kept_chunks
            llm_time = 0
            summary_time = 0
        
        # FAISS 인덱스 생성
        if self.method not in ["hipporag"]:
            print("\nCreating FAISS index...")
            start_faiss = time.time()
            
            # 임베딩 생성
            print("Generating embeddings...")
            if self.method == "naive_p":
                # 요약된 텍스트와 원본 텍스트를 구분하여 처리
                original_chunks = [item["chunk"] for item in kept]
                summarized_chunks = [item["summarized"] for item in summarized_final]
                
                # 원본 청크의 임베딩 선택
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                original_indices = [chunk_to_idx[chunk] for chunk in original_chunks]
                original_embeddings = chunk_embeddings[original_indices]
                
                # 요약된 텍스트의 임베딩 새로 생성
                if summarized_chunks:
                    summarized_embeddings = self.embed_texts(summarized_chunks)
                    # 모든 임베딩 결합
                    embeddings = np.vstack([original_embeddings, summarized_embeddings])
                else:
                    embeddings = original_embeddings
            else:
                # standard, random, cosine_only 방식의 경우
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                selected_indices = [chunk_to_idx[chunk] for chunk in merged_chunks]
                embeddings = chunk_embeddings[selected_indices]
            
            # 임베딩 정규화
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print(f"Generated {len(embeddings)} embeddings")
            
            dim = embeddings.shape[1]
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            index.add(embeddings)
            
            # 결과 저장
            print("Saving results...")
            faiss.write_index(index, self.index_file)
            np.save(self.embeddings_file, embeddings)
            self.save_jsonl(os.path.join(method_dir, "kept.jsonl"), [{"text": chunk} for chunk in merged_chunks])
            
            faiss_time = time.time() - start_faiss
        else:
            faiss_time = 0
        total_time = time.time() - start_total
        
        # 리포트 작성
        print("\nGenerating report...")
        fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "llm_filtered", "summarized", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "llm_time(s)", "summary_time(s)", "faiss_time(s)", "total_time(s)"]
        row = {
            "method": self.method,
            "persona_index": f"{persona_index}" if self.method in ["naive_p", "cosine_only"] else "all",
            "cosine_kept": len(kept_chunks) if self.method in ["naive_p", "cosine_only"] else 0,
            "random_kept": len(kept_chunks) if self.method == "random" else 0,
            "llm_filtered": len(filtered) if self.method == "naive_p" else 0,
            "summarized": len(summarized) if self.method == "naive_p" else 0,
            "kept": len(kept) if self.method == "naive_p" else 0,
            "cosine_filter_time(s)": f"{filter_time:.2f}" if self.method in ["naive_p", "cosine_only"] else "0",
            "random_filter_time(s)": f"{filter_time:.2f}" if self.method == "random" else "0",
            "llm_time(s)": f"{llm_time:.2f}",
            "summary_time(s)": f"{summary_time:.2f}",
            "faiss_time(s)": f"{faiss_time:.2f}",
            "total_time(s)": f"{total_time:.2f}"
        }
        self.save_csv(os.path.join(self.output_dir, INDEXING_REPORT_FILE), fieldnames, row)
        
        if persona_index == -1:
            print(f"\n=== Completed indexing ===")
        else:
            print(f"\n=== Completed indexing for persona {persona_index} ===")
        print(f"Total time: {total_time:.2f} seconds")
        return method_dir
