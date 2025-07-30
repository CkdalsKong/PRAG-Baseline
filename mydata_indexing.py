import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, PROMPT_LLM_FILTERING, PROMPT_LLM_FILTERING_NP, PROMPT_LLM_FILTERING_WOSUM, PROMPT_LLM_FILTERING_WOKEEP, PROMPT_LLM_FILTERING_SYSTEM, PROMPT_LLM_FILTERING_SYSTEM_EX, PROMPT_LLM_FILTERING_USER, PROMPT_LLM_FILTERING_SYSTEM_WOSUM, PROMPT_LLM_FILTERING_SYSTEM_WOSUM_EX, PROMPT_LLM_FILTERING_SYSTEM_WOKEEP, PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EX, PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_LD, PROMPT_LLM_FILTERING_SYSTEM_WOFILTER, PROMPT_LLM_FILTERING_SYSTEM_WOFILTER_EX, PROMPT_LLM_FILTERING_SYSTEM_LD, PROMPT_LLM_FILTERING_SYSTEM_EXPLICIT_LD, PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EXPLICIT_LD, PROMPT_LLM_FILTERING_USER_WOSUM, PROMPT_LLM_FILTERING_USER_WOKEEP, PROMPT_LLM_FILTERING_USER_WOKEEP_LD, PROMPT_LLM_FILTERING_USER_WOFILTER, PROMPT_LLM_FILTERING_USER_LD, PROMPT_LLM_FILTERING_USER_EXPLICIT_LD, PROMPT_LLM_FILTERING_USER_WOKEEP_EXPLICIT_LD, PROMPT_LLM_SUMMARIZING, PROMPT_LLM_SUMMARIZING_ONLY, PROMPT_LLM_FILTERING_P_NP, PROMPT_LLM_SUMMARIZING_ONLY_WOPREF, PROMPT_LLM_SUMMARIZING_ENHANCED, PROMPT_LLM_SUMMARIZING_SYSTEM, PROMPT_LLM_SUMMARIZING_SYSTEM_LD, PROMPT_LLM_SUMMARIZING_SYSTEM_EXPLICIT_LD, PROMPT_LLM_SUMMARIZING_USER, PROMPT_LLM_SUMMARIZING_USER_LD, PROMPT_LLM_SUMMARIZING_USER_EXPLICIT_LD, INDEXING_REPORT_FILE, THRESHOLD
# from HippoRAG import HippoRAG

import random
from sklearn.cluster import KMeans

class MyDataIndexing:
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
    
    def run_indexing_with_cache(self, persona_index, cached_resources):
        """캐시된 리소스를 사용하는 새로운 indexing 방식"""
        if persona_index == -1:
            print(f"\n=== Starting indexing with method {self.method} (cached) ===")
        else:
            print(f"\n=== Starting indexing for persona {persona_index} with method {self.method} (cached) ===")
        
        # 출력 디렉토리 설정
        if self.method in ["standard", "random", "random_1", "random_01", "hipporag", "raptor"]:
            method_dir = os.path.join(self.output_dir, f"{self.method}")
        else:
            method_dir = os.path.join(self.output_dir, f"{self.method}/{persona_index}")

        embeddings_file = os.path.join(method_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        index_file = os.path.join(method_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

        os.makedirs(method_dir, exist_ok=True)
        print(f"Output directory: {method_dir}")
        
        if self.method == "raptor":
            wiki_doc = []
            with open("data/corpus/sampled_wiki_doc.jsonl", "r") as f:
                for line in f:
                    if line.strip():  # 빈 줄 건너뛰기
                        wiki_doc.append(json.loads(line))
            
            # 문서 텍스트들을 하나의 문자열로 결합
            combined_text = "\n\n".join([doc["text"] for doc in wiki_doc])
            
            start_time = time.time()
            self.utils.raptor.add_documents(combined_text)
            self.utils.raptor.save(os.path.join(self.output_dir, "raptor_tree"))
            total_time = time.time() - start_time

            print("\nGenerating report...")
            fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "cluster_kept", "llm_filtered", "summarized", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "cluster_filter_time(s)", "llm_time(s)", "summary_time(s)", "faiss_time(s)", "total_time(s)"]
            row = {
                "method": self.method,
                "persona_index": "all",
                "cosine_kept": 0,
                "random_kept": 0,
                "cluster_kept": 0,
                "llm_filtered": 0,
                "summarized": 0,
                "kept": 0,
                "cosine_filter_time(s)": "0",
                "random_filter_time(s)": "0",
                "cluster_filter_time(s)": "0",
                "llm_time(s)": "0",
                "summary_time(s)": "0",
                "faiss_time(s)": "0",
                "total_time(s)": f"{total_time:.2f}"
            }
            self.utils.save_csv(os.path.join(self.output_dir, INDEXING_REPORT_FILE), fieldnames, row)
            return method_dir
            
        # Persona 데이터 로드
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "score_p", "naive_p_all", "wosum_p", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", \
                           "enhanced_prompt_ex", "enhanced_prompt", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", \
                            "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "sum_embedding", "wokeep_like_emdir"]:
            persona = self.utils.load_persona_data(persona_index)
            print(f"Loaded persona data for index {persona_index}")
        
        # 캐시된 리소스 사용 (모델 로딩 생략)
        print("✅ Using cached resources (models, chunks, embeddings)")
        chunks = cached_resources["chunks"]
        chunk_embeddings = cached_resources["embeddings"]
        print(f"Using {len(chunks)} cached chunks and their embeddings")
        
        # Preference embeddings 생성 또는 로드
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "score_p", "naive_p_all", "wosum_p", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", \
                           "enhanced_prompt_ex", "enhanced_prompt", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", \
                            "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "sum_embedding", "wokeep_like_emdir"]:
            print("Loading or generating preference embeddings...")
            preference_list = [block["preference"] for block in persona["preference_blocks"]]
            preference_emb_file = os.path.join(method_dir, "preference_embeddings.npy")
            
            if os.path.exists(preference_emb_file):
                print("Loading existing preference embeddings...")
                preference_embeddings = np.load(preference_emb_file)
            else:
                print("Generating new preference embeddings...")
                preference_embeddings = self.utils.embed_texts(preference_list)
                preference_embeddings = preference_embeddings / np.linalg.norm(preference_embeddings, axis=1, keepdims=True)
                np.save(preference_emb_file, preference_embeddings)
                print(f"Saved preference embeddings to {preference_emb_file}")
            
            # Preference 선호/불호 분리
            if self.method in ["score_p", "pref_cluster_filter", "llm_p_np", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_like_emdir"]:
                processed_preferences = []
                for preference in preference_list:
                    likes_dislikes = self.utils.extract_likes_dislikes(preference)
                    processed_preferences.append({
                        "likes": likes_dislikes["likes"],
                        "dislikes": likes_dislikes["dislikes"],
                        "original_preference": preference
                    })
                
                # 선호/불호 분리 정보 저장
                pref_info_file = os.path.join(method_dir, "preference_info.jsonl")
                self.utils.save_jsonl(pref_info_file, processed_preferences)
                print(f"✅ Preference info saved to {pref_info_file}")
                
                if self.method == "wokeep_like_embedding":
                    like_emb_file = os.path.join(method_dir, "like_embeddings.npy")
                    print("Generating like embeddings...")
                    like_embeddings = self.utils.embed_texts(processed_preferences["likes"])
                    like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
                    np.save(like_emb_file, like_embeddings)
                    print(f"Saved like embeddings to {like_emb_file}")

            chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            print(f"Using {len(preference_list)} preference embeddings")
        
        start_total = time.time()

        # 1차 필터링 : Cosine similarity 기반 필터링
        if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "wosum_p", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", \
                           "enhanced_prompt_ex", "enhanced_prompt", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", \
                            "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
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
        
        elif self.method in ["random", "random_1", "random_01"]:
            print("\nStarting random filtering...")
            # 전체 청크에서 10% 랜덤 샘플링
            if self.method == "random":
                num_samples = int(len(chunks) * 0.1)
            elif self.method == "random_1":
                num_samples = int(len(chunks) * 0.01)
            elif self.method == "random_01":
                num_samples = int(len(chunks) * 0.001)
            
            # numpy를 사용한 빠른 랜덤 샘플링
            indices = np.random.choice(len(chunks), size=num_samples, replace=False)
            kept_chunks = [chunks[i] for i in indices]
            kept_save = [{"chunk": chunk} for chunk in kept_chunks]
            
            # filtered_save는 나중에 필요할 때 생성
            filtered_save = []
            
            filter_time = time.time() - start_total
            print(f"Random sampling completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
            
        elif self.method == "score_p":
            print("\nStarting score filtering...")
            kept_save, kept_chunks, filtered_save = [], [], []
            keep_indices = []
            
            # 원본 preference 텍스트와 선호/불호 분리
            original_preferences = [block["preference"] for block in persona["preference_blocks"]]
            like_texts = [p["likes"] for p in processed_preferences]
            dislike_texts = [p["dislikes"] for p in processed_preferences]
            
            # 원본 preference와 선호/불호 임베딩 생성
            original_embeddings = self.utils.embed_texts(original_preferences)
            like_embeddings = self.utils.embed_texts(like_texts)
            dislike_embeddings = self.utils.embed_texts(dislike_texts)
            
            # 임베딩 정규화
            original_embeddings = original_embeddings / np.linalg.norm(original_embeddings, axis=1, keepdims=True)
            like_embeddings = like_embeddings / np.linalg.norm(like_embeddings, axis=1, keepdims=True)
            dislike_embeddings = dislike_embeddings / np.linalg.norm(dislike_embeddings, axis=1, keepdims=True)
            
            # 가중치 설정 (선호와 불호의 중요도 조절)
            alpha = 1.0  # 선호 가중치 (증가)
            beta = 1.0   # 불호 가중치
            epsilon = 1e-6  # 0으로 나누기 방지
            
            # THRESHOLD 설정
            # 스코어 범위: [-2.2, 2.2] (alpha=1.5, beta=1.0 기준)
            # 0.3: 보수적인 필터링 (선호가 불호보다 확실히 높은 경우만 선택)
            # 0.0: 중간 정도의 필터링 (선호가 불호보다 약간 높은 경우도 선택)
            # -0.2: 관대한 필터링 (불호가 선호보다 약간 높은 경우도 선택)
            SCORE_THRESHOLD = 0
            LIKE_THRESHOLD = 0.5  # 선호 유사도 최소 임계값
            ORIGINAL_THRESHOLD = 0.5  # 원본 preference 유사도 임계값
            
            # 배치 단위로 처리
            batch_size = self.batch_size
            for i in tqdm(range(0, len(chunk_embeddings_norm), batch_size), desc=f"Filtering persona {persona_index}"):
                batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                
                # 원본 preference, 선호와 불호에 대한 유사도 계산
                original_sims = np.dot(original_embeddings, batch_embeddings.T)
                like_sims = np.dot(like_embeddings, batch_embeddings.T)
                dislike_sims = np.dot(dislike_embeddings, batch_embeddings.T)
                
                # 각 preference별로 스코어 계산
                preference_scores = []
                for pref_idx, (original_sim, like_sim, dislike_sim) in enumerate(zip(original_sims, like_sims, dislike_sims)):
                    # dislikes가 "None"인 경우 원본 preference로만 판단
                    if dislike_texts[pref_idx] == "None":
                        # 원본 preference 유사도가 임계값을 넘는 경우만 고려
                        if np.any(original_sim > ORIGINAL_THRESHOLD):
                            pref_score = np.max(original_sim)  # 원본 preference 유사도만 사용
                        else:
                            pref_score = 0
                    else:
                        # 선호 유사도가 임계값을 넘는 경우만 고려
                        if np.any(like_sim > LIKE_THRESHOLD):
                            # 선호-불호 차이로 스코어 계산
                            pref_score = alpha * np.max(like_sim) - beta * np.max(dislike_sim)
                        else:
                            pref_score = 0
                    preference_scores.append(pref_score)
                
                # 최종 스코어
                preference_scores = np.array(preference_scores)  # 리스트를 numpy 배열로 변환
                scores = np.max(preference_scores, axis=0)  # 각 문서별 최대 스코어
                
                # 스코어가 임계값보다 큰 경우만 유지
                if np.isscalar(scores):
                    mask = [scores > SCORE_THRESHOLD] * len(batch_chunks)
                else:
                    mask = scores > SCORE_THRESHOLD
                
                # 결과 분류
                for j, (chunk, is_kept) in enumerate(zip(batch_chunks, mask)):
                    if is_kept:
                        kept_chunks.append(chunk)
                        kept_save.append({"chunk": chunk})
                    else:
                        filtered_save.append({"chunk": chunk})
            
            filter_time = time.time() - start_total
            print(f"Score-based filtering completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
     
        else:  # standard 방식
            filter_time = 0
            kept_chunks = chunks
            print("Skipping cosine filtering for standard method")
        
        # 새로운 pref_cluster_filter 방식 처리
        if self.method == "pref_cluster_filter":
            print("\nStarting preference-based clustering and filtering...")
            cluster_chunks = []
            
            start_clustering = time.time()
            
            # 2. 불호에 대한 필터링
            all_dislikes = [p["dislikes"] for p in processed_preferences]
            dislike_embeddings = self.utils.embed_texts(all_dislikes)
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
            for i in tqdm(range(0, len(kept_embeddings), batch_size), desc="Dislike-based filtering", leave=False, ncols=100):
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
            like_embeddings = self.utils.embed_texts(all_likes)
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
            self.utils.save_json(cluster_info_file, cluster_info)
            cluster_filter_time = time.time() - start_clustering

        # 2차 필터링: LLM 필터링
        if self.method in ["naive_p", "score_p", "naive_p_all", "wosum_p", "wokeep_p", "naive_p_np", "llm_p_np", "enhanced_sum", "per_pref", \
                           "enhanced_prompt_ex", "enhanced_prompt", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", \
                           "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
            print("\nStarting LLM filtering...")
            
            # 기존 결과 파일 확인
            result_info_file = os.path.join(method_dir, "result_info.jsonl")
            summarized_file = os.path.join(method_dir, "summarized.jsonl")
            
            # result_info.jsonl이 있으면 완료 여부 확인
            if os.path.exists(result_info_file):
                print(f"✅ LLM filtering already completed")
                # 기존 결과를 results 변수에 로드
                results = self.utils.load_json(result_info_file)
                print(f"📂 Loaded {len(results)} existing results from {result_info_file}")
                llm_time = 0  # 이미 완료된 작업이므로 시간 0으로 설정
                need_additional_processing = False
            else:
                need_additional_processing = True
                    
            if need_additional_processing:
                start_llm = time.time()
                if self.method == "wosum_p":
                    filtering_prompt = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_WOSUM)
                elif self.method == "wokeep_p":
                    filtering_prompt = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_WOKEEP)
                elif self.method == "naive_p_np":
                    filtering_prompt = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_NP)
                elif self.method == "llm_p_np":
                    filtering_prompt = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_P_NP)
                elif self.method in ["enhanced_prompt", "em_dir"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER)
                elif self.method in ["enhanced_prompt_ex", "em_dir_ex"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_EX)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER)
                elif self.method == "enhanced_prompt_wosum":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOSUM)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOSUM)
                elif self.method == "enhanced_prompt_wosum_ex":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOSUM_EX)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOSUM)
                elif self.method in ["enhanced_prompt_wokeep", "wokeep_emdir", "wokeep_like_emdir"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOKEEP)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOKEEP)
                elif self.method in ["enhanced_prompt_wokeep_ex", "wokeep_emdir_ex"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EX)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOKEEP)
                elif self.method == "enhanced_prompt_wofilter":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOFILTER)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOFILTER)
                elif self.method == "enhanced_prompt_wofilter_ex":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOFILTER_EX)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOFILTER)
                elif self.method == "like_dislike":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_LD)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_LD)
                elif self.method in ["wokeep_ld", "wokeep_ld_emdir"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_LD)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOKEEP_LD)
                elif self.method == "explicit_ld":
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_EXPLICIT_LD)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_EXPLICIT_LD)
                elif self.method in ["wokeep_explicit_ld", "wokeep_explicitld_emdir"]:
                    filtering_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EXPLICIT_LD)
                    filtering_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_FILTERING_USER_WOKEEP_EXPLICIT_LD)
                else:
                    # Prompt 로드
                    filtering_prompt = self.utils.load_prompt_template(PROMPT_LLM_FILTERING)

                # preference_list를 문자열로 변환
                preference_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(preference_list)])

                # if self.method == "llm_p_np":
                #     pos_preference = [p["likes"] for p in processed_preferences]
                #     neg_preference = [p["dislikes"] for p in processed_preferences]
                #     pos_text = "\n    - " + "\n    - ".join(pos_preference) if pos_preference else "\n    - None"
                #     neg_text = "\n    - " + "\n    - ".join(neg_preference) if neg_preference else "\n    - None"
                #     preference_text = (
                #         "- User Preference:\n"
                #         "  - Positive Preference:" + pos_text + "\n"
                #         "  - Negative Preference:" + neg_text
                #     )
                if self.method in ["explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir"]:
                    likes = [p["likes"] for p in processed_preferences]
                    dislikes = [p["dislikes"] for p in processed_preferences]
                    preference_text = "<likes>\n"
                    for i in range(5):
                        preference_text += f"{i + 1}. {likes[i]}\n"
                    preference_text += "</likes>\n\n<dislikes>\n"
                    for i in range(5):
                        preference_text += f"{i + 1}. {dislikes[i]}\n"
                    preference_text += "</dislikes>\n"


                start_llm = time.time()
                results = []
                if self.method == "per_pref":
                    with ThreadPoolExecutor() as executor:
                        futures = {executor.submit(self.utils.process_chunk_per_preference, kept_chunks[idx], preference_list, filtering_prompt): idx for idx in range(len(kept_chunks))}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                            result = future.result()
                            if result:
                                results.append(result)
                elif self.method in [""]:
                    # 첫 번째 청크로 실제 프롬프트 저장
                    if kept_chunks:
                        first_chunk = kept_chunks[0]
                        # 실제 LLM에 전달되는 프롬프트 생성
                        filled_user_prompt = filtering_prompt_user.format(preference=preference_text, chunk=first_chunk)
                        full_prompt = {
                            "system_prompt": filtering_prompt_system,
                            "user_prompt": filled_user_prompt,
                            "full_conversation": f"System: {filtering_prompt_system}\n\nUser: {filled_user_prompt}"
                        }
                        prompt_file = os.path.join(method_dir, "filtering_prompt_sample.json")
                        self.utils.save_json(prompt_file, full_prompt)
                        print(f"✅ Filtering prompt sample saved to {prompt_file}")
                    
                    # 실시간 저장 기능 사용
                    print(f"🔄 Using resume functionality for {self.method}")
                    
                    # 기존 결과 로드 (있다면)
                    existing_results = self.utils.load_existing_results_with_resume(result_info_file)
                    if existing_results:
                        print(f"📂 Loaded {len(existing_results)} existing results from {result_info_file}")
                    processed_indices = {result.get("chunk_index", -1) for result in existing_results}
                    
                    # 아직 처리되지 않은 청크들만 처리
                    remaining_chunks = [(idx, chunk) for idx, chunk in enumerate(kept_chunks) if idx not in processed_indices]
                    
                    if remaining_chunks:
                        print(f"📊 Processing {len(remaining_chunks)} remaining chunks out of {len(kept_chunks)} total")
                        with ThreadPoolExecutor() as executor:
                            futures = {executor.submit(self.utils.process_chunk_with_resume, chunk, preference_text, filtering_prompt_user, filtering_prompt_system, preference_list, result_info_file, idx): idx for idx, chunk in remaining_chunks}
                            for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                                result = future.result()
                                if result:
                                    results.append(result)
                    else:
                        print("✅ All chunks already processed!")
                    
                    # 최종 결과 로드 (실시간 저장된 것들)
                    results = self.utils.load_existing_results_with_resume(result_info_file)
                    
                elif self.method in ["enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
                    # 첫 번째 청크로 실제 프롬프트 저장
                    if kept_chunks:
                        first_chunk = kept_chunks[0]
                        # 실제 LLM에 전달되는 프롬프트 생성
                        filled_user_prompt = filtering_prompt_user.format(preference=preference_text, chunk=first_chunk)
                        full_prompt = {
                            "system_prompt": filtering_prompt_system,
                            "user_prompt": filled_user_prompt,
                            "full_conversation": f"System: {filtering_prompt_system}\n\nUser: {filled_user_prompt}"
                        }
                        prompt_file = os.path.join(method_dir, "filtering_prompt_sample.json")
                        self.utils.save_json(prompt_file, full_prompt)
                        print(f"✅ Filtering prompt sample saved to {prompt_file}")
                    
                    with ThreadPoolExecutor() as executor:
                        futures = {executor.submit(self.utils.process_chunk, kept_chunks[idx], preference_text, filtering_prompt_user, filtering_prompt_system, preference_list): idx for idx in range(len(kept_chunks))}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                            result = future.result()
                            if result:
                                results.append(result)
                else:
                    with ThreadPoolExecutor() as executor:
                        futures = {executor.submit(self.utils.process_chunk, kept_chunks[idx], preference_text, filtering_prompt, None, preference_list): idx for idx in range(len(kept_chunks))}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                            result = future.result()
                            if result:
                                results.append(result)

                result_info_file = os.path.join(method_dir, "result_info.jsonl")
                
                # enhanced_prompt와 enhanced_prompt_ex는 이미 실시간 저장되어 있음
                if self.method not in [""]:
                    self.utils.save_json(result_info_file, results)
                    print(f"✅ Result info saved to {result_info_file}")
                else:
                    print(f"✅ Result info already saved via resume functionality")
                
                llm_time = time.time() - start_llm

            # 결과 분류
            filtered, summarized, kept = [], [], []
            failed_chunks = []  # 실패한 청크들
            success_count = 0
            failed_count = 0

            for item in results:
                if item["status"] == "failed":
                    failed_chunks.append(item)
                    failed_count += 1
                    # 실패한 경우 기본적으로 필터
                    filtered.append({"chunk": item["chunk"]})
                else:
                    success_count += 1
                    if item["decision"] == "Filter":
                        filtered.append({"chunk": item["chunk"]})
                    elif item["decision"] == "Summarize":
                        if self.method == "per_pref":
                            summarized.append({
                                "chunk": item["chunk"],
                                "reason": item["reason"],
                                "relevant_preferences": item["relevant_preferences"]
                            })
                        elif self.method in ["enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
                            summarized.append({
                                "chunk": item["chunk"],
                                "reason": item["reason"],
                                "relevant_preference": item["relevant_preference"]
                            })
                        else:
                            summarized.append({
                                "chunk": item["chunk"],
                                "reason": item["reason"]
                            })
                    elif item["decision"] == "Keep As-Is":
                        if self.method in ["enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
                            kept.append({
                                "chunk": item["chunk"],
                                "relevant_preference": item.get("relevant_preference", "")
                            })
                        else:
                            kept.append({"chunk": item["chunk"]})
            
            print(f"LLM filtering completed. Success: {success_count}, Failed: {failed_count}")
            print(f"Results - Filtered: {len(filtered)}, Summarized: {len(summarized)}, Kept: {len(kept)}")
            
            # 실패한 청크들 저장
            if failed_chunks:
                failed_file = os.path.join(method_dir, "failed_chunks.jsonl")
                self.utils.save_jsonl(failed_file, failed_chunks)
                print(f"⚠️ Failed chunks saved to {failed_file}")
            
            # Summarization
            if self.method in ["wosum_p", "enhance_prompt_wosum", "enhance_prompt_wosum_ex"]:
                merged_chunks = [item["chunk"] for item in kept]
                summarized_final = []
                summary_time = 0
            else:
                print("\nStarting summarization...")
                
                # summarized.jsonl이 있으면 로드
                if os.path.exists(summarized_file):
                    print(f"Found existing summarized.jsonl, loading previous summarization results...")
                    summarized_final = []
                    
                    # summarized.jsonl은 JSONL 형식 (한 줄에 하나의 JSON 객체)
                    with open(summarized_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:  # 빈 줄 건너뛰기
                                try:
                                    summarized_final.append(json.loads(line))
                                except json.JSONDecodeError as e:
                                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                                    print(f"Line content: {line[:100]}...")  # 처음 100자만 출력
                                    continue
                    
                    print(f"Loaded {len(summarized_final)} previous summarization results")
                    summary_time = 0  # 이미 완료된 작업이므로 시간 0으로 설정
                else:
                    preference_text = "\n".join([f"- {p}" for p in preference_list])
                    # Prompt 로드
                    if self.method in ["enhanced_sum", "per_pref"]:
                        summarizing_prompt = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_ENHANCED)
                    elif self.method in ["enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_like_emdir"]:
                        summarizing_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_SYSTEM)
                        summarizing_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_USER)
                    elif self.method in ["like_dislike", "wokeep_ld", "wokeep_ld_emdir"]:
                        summarizing_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_SYSTEM_LD)
                        summarizing_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_USER_LD)
                    elif self.method in ["explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir"]:
                        summarizing_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_SYSTEM_EXPLICIT_LD)
                        summarizing_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_USER_EXPLICIT_LD)
                    else:
                        summarizing_prompt = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING)
                    
                    start_summary = time.time()
                    summarized_final = []
                    if self.method in ["enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
                        # 첫 번째 요약 대상 청크로 실제 프롬프트 저장
                        if summarized:
                            first_entry = summarized[0]
                            first_chunk = first_entry["chunk"]
                            # preference_text는 summarization용으로 다시 포맷팅
                            preference_text_sum = "\n".join([f"- {p}" for p in preference_list])

                            if self.method in ["explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir"]:
                                likes = [p["likes"] for p in processed_preferences]
                                dislikes = [p["dislikes"] for p in processed_preferences]
                                preference_text_sum = "<likes>\n"
                                for i in range(5):
                                    preference_text_sum += f"{i + 1}. {likes[i]}\n"
                                preference_text_sum += "</likes>\n\n<dislikes>\n"
                                for i in range(5):
                                    preference_text_sum += f"{i + 1}. {dislikes[i]}\n"
                                preference_text_sum += "</dislikes>\n"

                            # 실제 LLM에 전달되는 프롬프트 생성
                            filled_user_prompt = summarizing_prompt_user.format(preference=preference_text_sum, chunk=first_chunk, reason=first_entry["reason"])
                            full_prompt = {
                                "system_prompt": summarizing_prompt_system,
                                "user_prompt": filled_user_prompt,
                                "full_conversation": f"System: {summarizing_prompt_system}\n\nUser: {filled_user_prompt}"
                            }
                            prompt_file = os.path.join(method_dir, "summarizing_prompt_sample.json")
                            self.utils.save_json(prompt_file, full_prompt)
                            print(f"✅ Summarizing prompt sample saved to {prompt_file}")
                        
                        with ThreadPoolExecutor() as executor:  
                            futures = [executor.submit(self.utils.summarize_single, entry, summarizing_prompt_user, summarizing_prompt_system=summarizing_prompt_system) for entry in summarized]
                            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing chunks", leave=False, ncols=100):
                                try:
                                    result = future.result()
                                    summarized_final.append(result)
                                except Exception as e:
                                    print(f"Summarization failed: {e}")
                    else:
                        with ThreadPoolExecutor() as executor:  
                            futures = [executor.submit(self.utils.summarize_single, entry, summarizing_prompt, preference_text) for entry in summarized]
                            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing chunks", leave=False, ncols=100):
                                try:
                                    result = future.result()
                                    summarized_final.append(result)
                                except Exception as e:
                                    print(f"Summarization failed: {e}")
                    summary_time = time.time() - start_summary
                    self.utils.save_jsonl(summarized_file, summarized_final)
                    print(f"✅ Summary info saved to {summarized_file}")
                
                print(f"Summarization completed. Summarized {len(summarized_final)} chunks")
                
                # 최종 청크 병합 (요약된 텍스트만 사용)
                merged_chunks = [item["summarized"] for item in summarized_final] + [item["chunk"] for item in kept]
        else:
            merged_chunks = kept_chunks
            llm_time = 0
            summary_time = 0
        
        # ! sum_embedding summarization    
        if self.method == "sum_embedding":

            kept = []
            result_info_file = os.path.join(method_dir.replace("sum_embedding", "enhanced_prompt_wosum"), "result_info.jsonl")
            summarized_file = os.path.join(method_dir, "summarized.jsonl")
            with open(result_info_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
                for entry in entries:
                    if entry.get("decision") == "Keep As-Is":
                        kept.append(entry)

            preference_text = "\n".join([f"- {p}" for p in preference_list])
            # Prompt 로드
            summarizing_prompt_system = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_SYSTEM)
            summarizing_prompt_user = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_USER)
            
            start_summary = time.time()
            summarized_final = []
            if kept:
                first_entry = kept[0]
                first_chunk = first_entry["chunk"]
                # preference_text는 summarization용으로 다시 포맷팅
                preference_text_sum = "\n".join([f"- {p}" for p in preference_list])
                # 실제 LLM에 전달되는 프롬프트 생성
                filled_user_prompt = summarizing_prompt_user.format(preference=preference_text_sum, chunk=first_chunk, reason=first_entry["reason"])
                full_prompt = {
                    "system_prompt": summarizing_prompt_system,
                    "user_prompt": filled_user_prompt,
                    "full_conversation": f"System: {summarizing_prompt_system}\n\nUser: {filled_user_prompt}"
                }
                prompt_file = os.path.join(method_dir, "summarizing_prompt_sample.json")
                self.utils.save_json(prompt_file, full_prompt)
                print(f"✅ Summarizing prompt sample saved to {prompt_file}")
            
            with ThreadPoolExecutor() as executor:  
                futures = [executor.submit(self.utils.summarize_single, entry, summarizing_prompt_user, summarizing_prompt_system=summarizing_prompt_system) for entry in kept]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing chunks", leave=False, ncols=100):
                    try:
                        result = future.result()
                        summarized_final.append(result)
                    except Exception as e:
                        print(f"Summarization failed: {e}")
            summary_time = time.time() - start_summary
            self.utils.save_jsonl(summarized_file, summarized_final)
            print(f"✅ Summary info saved to {summarized_file}")
        
            print(f"Summarization completed. Summarized {len(summarized_final)} chunks")
            
            # 최종 청크 병합 (원본 텍스트만 사용)
            merged_chunks = [item["original"] for item in summarized_final]

        if self.method in ["sum_only_p", "sum_only_wopref"]:
            summarized_file = os.path.join(method_dir, "summarized.jsonl")
            print("\nStarting summarization...")
            
            # summarized.jsonl이 있으면 로드
            if os.path.exists(summarized_file):
                print(f"Found existing summarized.jsonl, loading previous summarization results...")
                summarized_final = []
                
                # summarized.jsonl은 JSONL 형식 (한 줄에 하나의 JSON 객체)
                with open(summarized_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 빈 줄 건너뛰기
                            try:
                                summarized_final.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                                print(f"Line content: {line[:100]}...")  # 처음 100자만 출력
                                continue
                
                print(f"Loaded {len(summarized_final)} previous summarization results")
                summary_time = 0  # 이미 완료된 작업이므로 시간 0으로 설정
            else:
                preference_text = "\n".join([f"- {p}" for p in preference_list])
                # Prompt 로드
                if self.method == "sum_only_p":
                    summarizing_prompt = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_ONLY)
                elif self.method == "sum_only_wopref":
                    summarizing_prompt = self.utils.load_prompt_template(PROMPT_LLM_SUMMARIZING_ONLY_WOPREF)
                
                start_summary = time.time()
                summarized_final = []
                with ThreadPoolExecutor() as executor:  
                    if self.method == "sum_only_p":
                        futures = [executor.submit(self.utils.summarize_single, entry, summarizing_prompt, preference_text) for entry in kept_chunks]
                    elif self.method == "sum_only_wopref":
                        futures = [executor.submit(self.utils.summarize_single, entry, summarizing_prompt) for entry in kept_chunks]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing chunks", leave=False, ncols=100):
                        try:
                            result = future.result()
                            summarized_final.append(result)
                        except Exception as e:
                            print(f"Summarization failed: {e}")
                summary_time = time.time() - start_summary
                self.utils.save_jsonl(summarized_file, summarized_final)
                print(f"✅ Summary info saved to {summarized_file}")
            
            print(f"Summarization completed. Summarized {len(summarized_final)} chunks")
            merged_chunks = [item["summarized"] for item in summarized_final]

        # FAISS 인덱스 생성
        if self.method not in ["hipporag"]:
            print("\nCreating FAISS index...")
            start_faiss = time.time()
            
            # 임베딩 생성
            print("Generating embeddings...")
            if self.method in ["em_dir", "em_dir_ex", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"]:
                # em_dir 방법: 임베딩에 선호 방향을 더해서 조정
                original_chunks = [item["chunk"] for item in kept]
                summarized_chunks = [item["summarized"] for item in summarized_final]
                
                # 원본 청크의 임베딩 선택
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                original_indices = [chunk_to_idx[chunk] for chunk in original_chunks]
                original_embeddings = chunk_embeddings[original_indices]
                
                # 요약된 텍스트의 임베딩 새로 생성
                if summarized_chunks:
                    summarized_embeddings = self.utils.embed_texts(summarized_chunks)
                    # 모든 임베딩 결합
                    base_embeddings = np.vstack([original_embeddings, summarized_embeddings])
                else:
                    base_embeddings = original_embeddings
                
                # 문서별 선호 매핑 정보 준비
                document_preference_mapping = []
                document_preference_mapping.extend(kept)
                document_preference_mapping.extend(summarized_final)
                
                # 선호 임베딩과 함께 임베딩 향상
                if self.method == "wokeep_like_embedding":
                    embeddings = self.utils.enhance_embeddings_with_preferences(
                        embeddings=base_embeddings,
                        preference_embeddings=like_embeddings,
                        document_preference_mapping=document_preference_mapping,
                        preference_list=preference_list
                    )
                else:
                    embeddings = self.utils.enhance_embeddings_with_preferences(
                        embeddings=base_embeddings,
                        preference_embeddings=preference_embeddings,
                        document_preference_mapping=document_preference_mapping,
                        preference_list=preference_list
                    )
                
                # 향상된 임베딩 정보 저장
                enhanced_info = {
                    "enhanced_count": len([d for d in document_preference_mapping if 'relevant_preference' in d and d['relevant_preference']]),
                    "total_count": len(document_preference_mapping),
                    "preference_mapping": document_preference_mapping
                }
                enhanced_info_file = os.path.join(method_dir, "enhanced_embedding_info.json")
                self.utils.save_json(enhanced_info_file, enhanced_info)
                print(f"✅ Enhanced embedding info saved to {enhanced_info_file}")
                
                # 원본 임베딩도 별도로 저장
                base_embeddings_file = os.path.join(method_dir, f"base_embeddings_{self.emb_model_name.replace('/', '_')}.npy")
                np.save(base_embeddings_file, base_embeddings)
                print(f"✅ Base embeddings saved to {base_embeddings_file}")
                
            elif self.method == "sum_embedding":
                summarized_chunks = [item["summarized"] for item in summarized_final]
                embeddings = self.utils.embed_texts(summarized_chunks)
                
            elif self.method in ["naive_p", "score_p", "naive_p_all", "wosum_p", "wokeep_p", "naive_p_np", "llm_p_np", "enhanced_sum", "per_pref", \
                               "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "like_dislike", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld"]:
                # 요약된 텍스트와 원본 텍스트를 구분하여 처리
                original_chunks = [item["chunk"] for item in kept]
                summarized_chunks = [item["summarized"] for item in summarized_final]
                
                # 원본 청크의 임베딩 선택
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                original_indices = [chunk_to_idx[chunk] for chunk in original_chunks]
                original_embeddings = chunk_embeddings[original_indices]
                
                # 요약된 텍스트의 임베딩 새로 생성
                if summarized_chunks:
                    summarized_embeddings = self.utils.embed_texts(summarized_chunks)
                    # 모든 임베딩 결합
                    embeddings = np.vstack([original_embeddings, summarized_embeddings])
                else:
                    embeddings = original_embeddings

            elif self.method in ["sum_only_p", "sum_only_wopref"]:
                # sum_only_p의 경우 요약된 텍스트의 임베딩을 새로 생성
                embeddings = self.utils.embed_texts(merged_chunks)

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
            elif self.method == "standard":
                embeddings = chunk_embeddings
            else:
                # standard, random, cosine_only 방식의 경우
                chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}
                selected_indices = [chunk_to_idx[chunk] for chunk in merged_chunks]
                embeddings = chunk_embeddings[selected_indices]
            
            # 임베딩 정규화
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print(f"Generated {len(embeddings)} embeddings")
            
            # 인덱스 타입에 따라 FAISS 인덱스 생성
            dim = embeddings.shape[1]
            if self.utils.index_type == "flat":
                print("Creating FAISS IndexFlatIP...")
                index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
            elif self.utils.index_type == "hnsw":
                print("Creating FAISS IndexHNSWFlat...")
                index = faiss.IndexHNSWFlat(dim, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64

            else:
                raise ValueError(f"Unsupported index type: {self.utils.index_type}")
            
            index.add(embeddings.astype(np.float32))
            
            # 결과 저장
            print("Saving results...")
            faiss.write_index(index, index_file)
            print(f"FAISS saved in {index_file}")
            np.save(embeddings_file, embeddings)
            self.utils.save_jsonl(os.path.join(method_dir, "kept.jsonl"), [{"text": chunk} for chunk in merged_chunks])
            
            faiss_time = time.time() - start_faiss
        else:
            faiss_time = 0
        total_time = time.time() - start_total
        
        # 리포트 작성
        print("\nGenerating report...")
        fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "cluster_kept", "llm_filtered", "summarized", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "cluster_filter_time(s)", "llm_time(s)", "summary_time(s)", "faiss_time(s)", "total_time(s)"]
        row = {
            "method": self.method,
            "persona_index": f"{persona_index}" if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "score_p", "naive_p_all", "wosum_p", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"] else "all",
            "cosine_kept": len(kept_chunks) if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "score_p", "naive_p_all", "wosum_p", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"] else 0,
            "random_kept": len(kept_chunks) if self.method in ["random", "random_1", "random_01"] else 0,
            "cluster_kept": sum(len(chunks) for chunks in cluster_chunks) if self.method == "pref_cluster_filter" else 0,
            "llm_filtered": len(filtered) if self.method in ["naive_p", "score_p", "naive_p_all", "wosum_p", "wokeep_p", "naive_p_np", "llm_p_np", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"] else 0,
            "summarized": len(summarized_final) if self.method in ["naive_p", "score_p", "naive_p_all", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"] else 0,
            "kept": len(kept) if self.method in ["naive_p", "score_p", "naive_p_all", "wosum_p", "naive_p_np", "llm_p_np", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "explicit_ld", "sum_embedding"] else 0,
            "cosine_filter_time(s)": f"{filter_time:.2f}" if self.method in ["naive_p", "cosine_only", "pref_cluster_filter", "score_p", "wosum_p", "naive_p_all", "sum_only_p", "wokeep_p", "naive_p_np", "llm_p_np", "sum_only_wopref", "enhanced_sum", "per_pref", "enhanced_prompt", "enhanced_prompt_ex", "enhanced_prompt_wosum", "enhanced_prompt_wosum_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wofilter_ex", "em_dir", "em_dir_ex", "like_dislike", "wokeep_emdir", "wokeep_emdir_ex", "wokeep_ld", "explicit_ld", "wokeep_explicit_ld", "wokeep_explicitld_emdir", "wokeep_ld_emdir", "wokeep_like_emdir"] else "0",
            "random_filter_time(s)": f"{filter_time:.2f}" if self.method in ["random", "random_1", "random_01"] else "0",
            "cluster_filter_time(s)": f"{cluster_filter_time:.2f}" if self.method == "pref_cluster_filter" else "0",
            "llm_time(s)": f"{llm_time:.2f}",
            "summary_time(s)": f"{summary_time:.2f}",
            "faiss_time(s)": f"{faiss_time:.2f}",
            "total_time(s)": f"{total_time:.2f}"
        }
        self.utils.save_csv(os.path.join(self.output_dir, INDEXING_REPORT_FILE), fieldnames, row)
        
        if persona_index == -1:
            print(f"\n=== Completed indexing ===")
        else:
            print(f"\n=== Completed indexing for persona {persona_index} ===")
        print(f"Total time: {total_time:.2f} seconds")
        return method_dir
