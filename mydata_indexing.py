import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, PROMPT_LLM_FILTERING, PROMPT_LLM_SUMMARIZING, INDEXING_REPORT_FILE, THRESHOLD, TOP_K
# from HippoRAG import HippoRAG

import random
from sklearn.cluster import KMeans

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

        embeddings_file = os.path.join(method_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        index_file = os.path.join(method_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

        os.makedirs(method_dir, exist_ok=True)
        print(f"Output directory: {method_dir}")
        
        # 데이터 로드
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"]:
            persona = self.load_persona_data(persona_index)
            print(f"Loaded persona data for index {persona_index}")
        
        self.load_models()

        # chunks와 embeddings 로드
        print("Loading chunks and embeddings...")
        with open(self.chunk_file, "r", encoding="utf-8") as f:
            chunks = [json.loads(line)["text"] for line in f]
        chunk_embeddings = np.load(self.embedding_file)
        print(f"Loaded {len(chunks)} chunks and their embeddings")
        
        # Preference embeddings 생성 또는 로드
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"]:
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
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"]:
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
        
        # 새로운 pref_cluster_filter 방식 처리
        if self.method == "pref_cluster_filter":
            print("\nStarting preference-based clustering and filtering...")
            cluster_chunks = []
            
            # 1. 선호도에서 호/불호 추출
            processed_preferences = []
            for block in persona["preference_blocks"]:
                preference = block["preference"]
                likes_dislikes = self.extract_likes_dislikes(preference)
                processed_preferences.append({
                    "likes": likes_dislikes["likes"],
                    "dislikes": likes_dislikes["dislikes"],
                    "original_preference": preference
                })
            start_clustering = time.time()
            # 2. 불호에 대한 필터링
            all_dislikes = [p["dislikes"] for p in processed_preferences]
            dislike_embeddings = self.embed_texts(all_dislikes)
            dislike_embeddings = dislike_embeddings / np.linalg.norm(dislike_embeddings, axis=1, keepdims=True)
            
            # 불호와 유사한 청크 제거
            filtered_chunks = []
            filtered_indices = []
            removed_by_dislike = 0  # 불호에 의해 제거된 청크 수
            

            # 원본 청크의 임베딩 선택
            chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
            kept_indices = [chunk_to_idx[chunk] for chunk in kept_chunks]
            kept_embeddings = chunk_embeddings[kept_indices]

            kept_embeddings = kept_embeddings / np.linalg.norm(kept_embeddings, axis=1, keepdims=True)
            
            batch_size = self.batch_size
            for i in tqdm(range(0, len(kept_embeddings), batch_size), desc="Dislike-based filtering"):
                batch_embeddings = kept_embeddings[i:i + batch_size]
                batch_chunks = [kept_chunks[j] for j in range(i, min(i + batch_size, len(kept_chunks)))]
                
                dislike_sims = np.dot(dislike_embeddings, batch_embeddings.T)
                dislike_mask = np.any(dislike_sims > THRESHOLD, axis=0)
                
                for j, (chunk, is_disliked) in enumerate(zip(batch_chunks, dislike_mask)):
                    if not is_disliked:
                        filtered_chunks.append(chunk)
                        filtered_indices.append(kept_indices[i + j])
                    else:
                        removed_by_dislike += 1

            # 3. 호에 대한 클러스터링
            all_likes = [p["likes"] for p in processed_preferences]
            like_embeddings = self.embed_texts(all_likes)
            like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
            
            # 필터링된 청크들의 임베딩
            filtered_chunk_embeddings = chunk_embeddings[filtered_indices]
            filtered_chunk_embeddings = filtered_chunk_embeddings / np.linalg.norm(filtered_chunk_embeddings, axis=1, keepdims=True)
            
            # 남은 청크 수에 따라 각 클러스터당 저장할 청크 수 결정
            n_filtered_chunks = len(filtered_chunks)
            chunks_per_cluster = min(5000, max(100, n_filtered_chunks // 5))  # 전체 청크의 1/5, 최소 20개, 최대 100개
            
            # 각 호에 대해 관련 청크 찾기
            cluster_chunks = []  # 각 클러스터별 청크 리스트
            for like_emb in like_embeddings:
                # 호와의 유사도 계산
                similarities = np.dot(filtered_chunk_embeddings, like_emb)
                # 상위 K개 청크 선택
                top_k_indices = np.argsort(similarities)[-chunks_per_cluster:]
                # 원본 chunks에서 선택
                cluster_chunks.append([chunks[filtered_indices[i]] for i in top_k_indices])
            
            # 4. 클러스터 정보 저장
            cluster_info = {
                "likes": all_likes,
                "dislikes": all_dislikes,
                "cluster_chunks": cluster_chunks,  # 각 호별 관련 청크들
                "removed_by_dislike": removed_by_dislike,  # 불호에 의해 제거된 청크 수
                "total_chunks": len(chunks),  # 전체 청크 수
                "filtered_chunks": len(filtered_chunks),  # 필터링 후 남은 청크 수
                "chunks_per_cluster": chunks_per_cluster  # 클러스터당 저장된 청크 수
            }
            cluster_info_file = os.path.join(method_dir, "cluster_info.json")
            self.save_json(cluster_info_file, cluster_info)
            cluster_filter_time = time.time() - start_clustering

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

            elif self.method == "pref_cluster_filter":
                # 클러스터링된 청크들만 사용
                all_cluster_chunks = []
                for chunk in cluster_chunks:
                    all_cluster_chunks.extend(chunk)
                
                # 원본 청크의 임베딩 선택
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                selected_indices = [chunk_to_idx[chunk] for chunk in all_cluster_chunks]
                    
                embeddings = chunk_embeddings[selected_indices]
                
                # merged_chunks 업데이트 (나중에 kept.jsonl 저장에 사용)
                merged_chunks = [chunks[i] for i in selected_indices]  # 원본 chunks에서 선택
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
            faiss.write_index(index, index_file)
            print(f"FAISS saved in {index_file}")
            np.save(embeddings_file, embeddings)
            self.save_jsonl(os.path.join(method_dir, "kept.jsonl"), [{"text": chunk} for chunk in merged_chunks])
            
            faiss_time = time.time() - start_faiss
        else:
            faiss_time = 0
        total_time = time.time() - start_total
        
        # 리포트 작성
        print("\nGenerating report...")
        fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "cluster_kept", "llm_filtered", "summarized", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "cluster_filter_time(s)", "llm_time(s)", "summary_time(s)", "faiss_time(s)", "total_time(s)"]
        row = {
            "method": self.method,
            "persona_index": f"{persona_index}" if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"] else "all",
            "cosine_kept": len(kept_chunks) if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"] else 0,
            "random_kept": len(kept_chunks) if self.method == "random" else 0,
            "cluster_kept": sum(len(chunks) for chunks in cluster_chunks) if self.method == "pref_cluster_filter" else 0,
            "llm_filtered": len(filtered) if self.method == "naive_p" else 0,
            "summarized": len(summarized) if self.method == "naive_p" else 0,
            "kept": len(kept) if self.method == "naive_p" else 0,
            "cosine_filter_time(s)": f"{filter_time:.2f}" if self.method in ["naive_p", "cosine_only", "pref_cluster_filter"] else "0",
            "random_filter_time(s)": f"{filter_time:.2f}" if self.method == "random" else "0",
            "cluster_filter_time(s)": f"{cluster_filter_time:.2f}" if self.method == "pref_cluster_filter" else "0",
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
