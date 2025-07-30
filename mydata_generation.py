import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, PROMPT_GENERATION, GENERATION_REPORT_FILE, TOP_K
from concurrent.futures import ProcessPoolExecutor
from raptor.RetrievalAugmentation import RetrievalAugmentation

class MyDataGeneration:
    def __init__(self, utils):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.use_multi_gpu = utils.use_multi_gpu
        self.chunk_mode = utils.chunk_mode
        self.output_dir = utils.output_dir
        self.persona_task_file = utils.persona_task_file
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.chunk_file = utils.chunk_file
        self.embedding_file = utils.embedding_file
        self.batch_size = utils.batch_size
        self.use_trust_align = utils.use_trust_align

    def process_query(self, query, preference_text, preferences, filtered_chunks, index, method_dir, generation_prompt):
        question = query["question"]

        if self.method == "pref_cluster_filter":
            # 1. 클러스터 정보 로드
            cluster_info_file = os.path.join(method_dir, "cluster_info.json")
            with open(cluster_info_file, 'r') as f:
                cluster_info = json.load(f)
            
            # 2. 쿼리 임베딩 생성
            query_emb = self.utils.embed_query(question)
            query_emb = query_emb / np.linalg.norm(query_emb)
            
            start_retrieval = time.time()

            # 3. 쿼리와 각 호의 유사도 계산
            likes = cluster_info["likes"]
            like_embeddings = self.utils.embed_texts(likes)
            like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
            
            # 쿼리 임베딩을 2차원으로 확장
            query_emb_2d = query_emb.reshape(1, -1)  # shape: (1, embedding_dim)

            # 명시적으로 행렬 곱셈 수행
            similarities = np.dot(like_embeddings, query_emb_2d.T).flatten()  # shape: (5,)
            best_cluster_idx = np.argmax(similarities)
            
            # 4. 선택된 클러스터의 청크들 중에서 쿼리와 가장 유사한 상위 K개 선택
            cluster_chunks = cluster_info["cluster_chunks"][best_cluster_idx]
            
            # 클러스터의 청크들에 대한 임베딩 생성
            cluster_chunk_embeddings = self.utils.embed_texts(cluster_chunks)
            cluster_chunk_embeddings = cluster_chunk_embeddings / np.linalg.norm(cluster_chunk_embeddings, axis=1, keepdims=True)
            
            # 쿼리와 청크들 간의 유사도 계산
            chunk_similarities = np.dot(cluster_chunk_embeddings, query_emb_2d.T).flatten()  # shape: (n_chunks,)
            
            # 상위 K개 청크 선택
            top_k_indices = np.argsort(chunk_similarities)[-TOP_K:]
            retrieved = [cluster_chunks[i] for i in top_k_indices]
            retrieval_time = time.time() - start_retrieval

        elif self.method[-2:] == "wq":
            """기존 cheating method"""
            retrieved, retrieval_time = self.utils.retrieve_top_k_wq(
                question,
                preference_text,
                index,
                filtered_chunks
            )
        
        elif self.method[-3:] == "wql":
            """다 넘겨준 후, LLM으로 관련 선호 inference"""
            retrieved, retrieval_time = self.utils.retrieve_top_k_wq_llm(
                question,
                preferences,
                index,
                filtered_chunks
            )

        else:
            # 기존 방식대로 처리
            retrieved, retrieval_time = self.utils.retrieve_top_k(
                question,
                index,
                filtered_chunks
            )
        
        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])

        # 프롬프트 생성
        filled_prompt = generation_prompt.replace("{context}", context).replace("{question}", question)
        
        # vLLM을 사용한 생성
        try:
            generated_text = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt="You are a helpful assistant for generating responses.",
                max_tokens=1024  # 생성 시 더 긴 텍스트 허용
            )
            
            return {
                "preference": preference_text,
                "question": question,
                "response_to_q": generated_text,
                "retrieved_docs": retrieved,
                "retrieval_time": retrieval_time
            }
        except Exception as e:
            print(f"Failed to generate response: {e}")
            return None
    
    def run_generation_with_cache(self, persona_index, method_dir, cached_resources):
        """캐시된 리소스를 사용하는 새로운 generation 방식"""
        print(f"\n=== Starting generation for persona {persona_index} with method {self.method} (cached) ===")

        # 인덱스 타입에 따라 인덱스 파일 경로 결정
        model_name_clean = self.emb_model_name.replace("/", "_")
        if self.utils.index_type == "flat":
            index_file = os.path.join(method_dir, f"index_flat_{model_name_clean}.faiss")
        elif self.method == "raptor":
            index_file = os.path.join(self.output_dir, "raptor_tree")
        else:
            index_file = os.path.join(method_dir, f"index_{model_name_clean}.faiss")

        # Index 로드
        if self.method not in ["raptor"]:
            index = faiss.read_index(index_file)

        # 데이터 로드
        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")
        
        all_results = []
        retrieval_times = []

        if self.method == "raptor":
            if self.utils.raptor is None:
                self.utils.raptor = RetrievalAugmentation(tree=index_file)
            for query in persona["queries"]:
                start_time = time.time()
                answer = self.utils.raptor.answer_question(query["question"])
                retrieval_times.append(time.time() - start_time)
                all_results.append({
                    "preference": query["preference"],
                    "question": query["question"],
                    "response_to_q": answer
                })
            output_file = os.path.join(method_dir, f"gen_raptor_{persona_index}.json")

            self.utils.save_json(output_file, all_results)
            print(f"✅ Generation results saved to {output_file}")

            # Retrieval 시간 통계 계산
            avg_time = np.mean(retrieval_times)
            max_time = np.max(retrieval_times)
            min_time = np.min(retrieval_times)
            
            # 리포트 작성
            fieldnames = ["method", "persona_index", "avg_retrieval_time(s)", "max_retrieval_time(s)", "min_retrieval_time(s)"]
            row = {
                "method": self.method,
                "persona_index": f"{persona_index}",
                "avg_retrieval_time(s)": f"{avg_time:.4f}",
                "max_retrieval_time(s)": f"{max_time:.4f}",
                "min_retrieval_time(s)": f"{min_time:.4f}"
            }
            
            self.utils.save_csv(os.path.join(self.output_dir, GENERATION_REPORT_FILE), fieldnames, row)
            return method_dir

        # 필터링된 청크 로드
        filtered_chunks_file = os.path.join(method_dir, "kept.jsonl")
        with open(filtered_chunks_file, "r", encoding="utf-8") as f:
            filtered_chunks = [json.loads(line)["text"] for line in f]
        print(f"Loaded {len(filtered_chunks)} filtered chunks")
        
        # Prompt 로드
        generation_prompt = self.utils.load_prompt_template(PROMPT_GENERATION)

        # 캐시된 모델 사용 (모델 로딩 생략)
        print("✅ Using cached models")


        
        # 각 preference block에 대해 처리
        for block in persona["preference_blocks"]:
            preference_text = block["preference"]
            preferences = [block["preference"] for block in persona["preference_blocks"]]
            queries = block["queries"]
            
            # 각 쿼리에 대해 ProcessPoolExecutor로 처리
            with ProcessPoolExecutor(max_workers=2) as executor:  # GPU 개수만큼 worker 설정
                futures = []
                for query in queries:
                    future = executor.submit(
                        self.process_query,
                        query,
                        preference_text,    # 기존 cheating method
                        preferences,        # 다 넘겨준 후, LLM으로 관련 선호 inference
                        filtered_chunks,
                        index,  # 인덱스 전달
                        method_dir,
                        generation_prompt
                    )
                    futures.append(future)
                
                # 결과 수집
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing queries for preference: {preference_text[:50]}..."):
                    result = future.result()
                    if result:
                        all_results.append(result)
                        retrieval_times.append(result["retrieval_time"])
                    
                    # 메모리 정리 강화
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 결과 저장
        if self.utils.index_type == "flat":
            if self.method[-2:] == "wq":
                output_file = os.path.join(method_dir, f"gen_wq_flat_{persona_index}.json")
            elif self.method[-3:] == "wql":
                output_file = os.path.join(method_dir, f"gen_wql_flat_{persona_index}.json")
            else:
                output_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")
        else:
            if self.method[-2:] == "wq":
                output_file = os.path.join(method_dir, f"gen_wq_{persona_index}.json")
            elif self.method[-3:] == "wql":
                output_file = os.path.join(method_dir, f"gen_wql_{persona_index}.json")
            elif self.use_trust_align:
                output_file = os.path.join(method_dir, f"gen_{self.method}_trustalign_{persona_index}.json")
            else:
                output_file = os.path.join(method_dir, f"gen_{self.method}_{persona_index}.json")
        self.utils.save_json(output_file, all_results)
        print(f"✅ Generation results saved to {output_file}")

        # Retrieval 시간 통계 계산
        avg_time = np.mean(retrieval_times)
        max_time = np.max(retrieval_times)
        min_time = np.min(retrieval_times)
        
        # 리포트 작성
        fieldnames = ["method", "persona_index", "avg_retrieval_time(s)", "max_retrieval_time(s)", "min_retrieval_time(s)"]
        row = {
            "method": self.method,
            "persona_index": f"{persona_index}",
            "avg_retrieval_time(s)": f"{avg_time:.4f}",
            "max_retrieval_time(s)": f"{max_time:.4f}",
            "min_retrieval_time(s)": f"{min_time:.4f}"
        }
        
        self.utils.save_csv(os.path.join(self.output_dir, GENERATION_REPORT_FILE), fieldnames, row)
        
        print(f"\n=== Completed generation for persona {persona_index} ===")
        print(f"Average retrieval time: {avg_time:.4f} seconds")
        print(f"Max retrieval time: {max_time:.4f} seconds")
        print(f"Min retrieval time: {min_time:.4f} seconds")
        return method_dir