import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, PROMPT_GENERATION, GENERATION_REPORT_FILE

class MyDataGeneration(MyDataUtils):
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
        
        # 모델 로드
        self.tokenizer, self.model = self.utils.load_models()
        
        # 임베딩 파일 경로 설정
        self.embeddings_file = os.path.join(self.output_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        self.index_file = os.path.join(self.output_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

    def process_query(self, query, preference_text, filtered_chunks, index, generation_prompt):
        question = query["question"]
        
        # 관련 청크 검색
        retrieved, retrieval_time = self.retrieve_top_k(
            question,
            index,
            filtered_chunks
        )
        
        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])

        # 프롬프트 생성
        filled_prompt = generation_prompt.replace("{context}", context).replace("{question}", question)
        
        # vLLM을 사용한 생성
        try:
            generated_text = self.generate_message_vllm(
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

    def run_generation(self, persona_index, method_dir):
        print(f"\n=== Starting generation for persona {persona_index} with method {self.method} ===")
        
        # 출력 디렉토리 설정
        if self.method in ["standard", "random"]:
            method_dir = os.path.join(self.output_dir, f"{self.method}")
        else:
            method_dir = os.path.join(self.output_dir, f"{self.method}/{persona_index}")
        os.makedirs(method_dir, exist_ok=True)

        # 데이터 로드
        persona = self.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")
        
        # 필터링된 청크 로드
        filtered_chunks_file = os.path.join(method_dir, "kept.jsonl")
        with open(filtered_chunks_file, "r", encoding="utf-8") as f:
            filtered_chunks = [json.loads(line)["text"] for line in f]
        print(f"Loaded {len(filtered_chunks)} filtered chunks")
        
        # Prompt 로드
        generation_prompt = self.load_prompt_template(PROMPT_GENERATION)
        
        # Index 로드
        index = faiss.read_index(os.path.join(method_dir, "faiss.index"))

        # Model 로드
        self.load_models()

        all_results = []
        retrieval_times = []
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # 각 preference block에 대해 처리
            for block in persona["preference_blocks"]:
                preference_text = block["preference"]
                queries = block["queries"]
                
                # 각 쿼리에 대해 future 생성
                for query in queries:
                    future = executor.submit(
                        self.process_query,
                        query,
                        preference_text,
                        filtered_chunks,
                        index,
                        generation_prompt
                    )
                    futures.append(future)
            
            # 결과 수집
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                result = future.result()
                if result:
                    all_results.append(result)
                    retrieval_times.append(result["retrieval_time"])
        
        # 결과 저장
        output_file = os.path.join(method_dir, f"gen_{self.method}_{persona_index}.json")
        self.save_json(output_file, all_results)
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
        
        self.save_csv(os.path.join(self.output_dir, GENERATION_REPORT_FILE), fieldnames, row)
        
        print(f"\n=== Completed generation for persona {persona_index} ===")
        print(f"Average retrieval time: {avg_time:.4f} seconds")
        print(f"Max retrieval time: {max_time:.4f} seconds")
        print(f"Min retrieval time: {min_time:.4f} seconds")
        return method_dir