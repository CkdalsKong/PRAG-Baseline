import os
import csv
import time
import json
import torch
import warnings
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue, cpu_count, set_start_method
from tqdm.auto import tqdm  # tqdm.auto를 사용하여 더 나은 진행 상황 표시
import threading
import requests
import random

# 토크나이저 병렬화 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 멀티프로세싱 시작 방식 설정
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # 이미 설정되어 있는 경우 무시

# 공통 설정
ROOT_DIR = "/data/my_PRAG/baseline"
PERSONA_TASK_FILE = os.path.join(ROOT_DIR, "final_persona_tasks.json")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VLLM_SERVER_URL = "http://localhost:8006/v1"
TOP_K = 5

# Prompt 템플릿 파일 경로
PROMPT_DIR = os.path.join(ROOT_DIR, "prompt")
PROMPT_LLM_FILTERING = os.path.join(PROMPT_DIR, "mydata_llm_filtering.txt")
PROMPT_LLM_SUMMARIZING = os.path.join(PROMPT_DIR, "mydata_llm_summarizing.txt")
PROMPT_GENERATION = os.path.join(PROMPT_DIR, "mydata_generation.txt")

# Indexing 관련 설정
INDEXING_REPORT_FILE = "indexing_report.csv"
THRESHOLD = 0.4

# Generation 관련 설정
GENERATION_REPORT_FILE = "generation_report.csv"

# Evaluation 관련 설정
EVALUATION_REPORT_FILE = "evaluation_report.csv"
ERROR_TYPE_DIR = os.path.join(ROOT_DIR, "error_type")

def set_global_seed(seed):
    """모든 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

class MyDataUtils:
    # 클래스 변수로 시드 설정
    _seed = 42
    set_global_seed(_seed)
    
    def __init__(self, mode="all", method="all", device="cuda:0", use_multi_gpu=False, chunk_mode="wodoc", output_dir=None):
        self.mode = mode
        self.method = method
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        
        # chunk_mode에 따른 파일 경로 설정
        if chunk_mode == "wodoc":
            self.chunk_file = os.path.join(ROOT_DIR, "corpus/sampled_chunks.jsonl")
            self.embedding_file = os.path.join(ROOT_DIR, "corpus/sampled_embeddings.npy")
        elif chunk_mode == "wdoc":
            self.chunk_file = os.path.join(ROOT_DIR, "corpus/sampled_chunks_with_doc.jsonl")
            self.embedding_file = os.path.join(ROOT_DIR, "corpus/sampled_embeddings_with_doc.npy")
        else:
            raise ValueError(f"Invalid chunk_mode: {chunk_mode}. Must be either 'wodoc' or 'wdoc'")
        
        self.output_dir = os.path.join(ROOT_DIR, output_dir)
        self.contriever_tokenizer = None
        self.contriever_model = None
        self.hf_tokenizer = None
        self.hf_model = None
        
        # GPU 메모리 최적화를 위한 배치 사이즈 설정
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            self.num_gpus = torch.cuda.device_count()
            # 각 GPU의 메모리 크기에 따라 배치 사이즈 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                self.base_batch_size = 64
            elif gpu_memory >= 48e9:  # 48GB
                self.base_batch_size = 128
            elif gpu_memory >= 24e9:  # 24GB
                self.base_batch_size = 64
            elif gpu_memory >= 16e9:  # 16GB
                self.base_batch_size = 32
            else:
                self.base_batch_size = 16
            self.batch_size = self.base_batch_size * self.num_gpus
        else:
            self.num_gpus = 1
            # 단일 GPU의 경우에도 메모리 크기에 따라 배치 사이즈 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                self.batch_size = 256
            elif gpu_memory >= 48e9:  # 48GB
                self.batch_size = 128
            elif gpu_memory >= 24e9:  # 24GB
                self.batch_size = 64
            elif gpu_memory >= 16e9:  # 16GB
                self.batch_size = 32
            else:
                self.batch_size = 16
    
    def load_models(self):
        """모델과 토크나이저 로드"""
        # Contriever 모델 로드
        print("Loading Contriever model...")
        self.contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.contriever_model = AutoModel.from_pretrained("facebook/contriever").eval()
        
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {self.num_gpus} GPUs with batch size {self.batch_size}!")
            # 먼저 모델을 device로 이동
            self.contriever_model = self.contriever_model.to(self.device)
            # 그 다음 DataParallel 적용
            self.contriever_model = DataParallel(
                self.contriever_model,
                device_ids=list(range(self.num_gpus)),
                output_device=0,  # 메인 GPU
                dim=0  # 배치 차원
            )
            print(f"Contriever model distributed across {self.num_gpus} GPUs")
        else:
            self.contriever_model = self.contriever_model.to(self.device)
            print(f"Contriever model loaded on {self.device}")
        
        # LLM 모델 로드 (필요한 경우)
        if self.method in ["naive_p"] and self.mode in [""]:
            print("Loading LLM model...")
            set_seed(42)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.hf_tokenizer.padding_side = 'left'  # 왼쪽 패딩으로 설정
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            # GPU 메모리 관리를 위한 설정
            torch.cuda.empty_cache()
            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                # 각 GPU에 모델을 분산하여 로드
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",  # 자동으로 GPU에 분산
                    use_cache=True,     # 캐시 사용
                    low_cpu_mem_usage=True  # CPU 메모리 사용 최소화
                )
                print(f"LLM model automatically distributed across {torch.cuda.device_count()} GPUs")
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    use_cache=True,
                    low_cpu_mem_usage=True
                ).to(self.device)
                print(f"LLM model loaded on {self.device}")
            self.hf_model.eval()
    
    def embed_texts(self, texts):
        all_embs = []
        # 배치 사이즈 최적화
        batch_size = self.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.contriever_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.contriever_model(**inputs)
                if isinstance(outputs.last_hidden_state, tuple):
                    embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(embeddings)
        return np.vstack(all_embs)
    
    def embed_query(self, query):
        inputs = self.contriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.contriever_model(**inputs)
            if isinstance(outputs.last_hidden_state, tuple):
                query_emb = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
            else:
                query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        return query_emb
    
    def call_llm(self, prompts, max_new_tokens=512):
        """LLM 호출 (배치 처리)"""
        try:
            # 배치 단위로 처리
            results = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Starting from v4.46, the `logits` model output will have the same type as the model",
                        category=UserWarning,
                    )
                    # 배치 전체를 한 번에 처리
                    formatted_prompts = [
                        self.hf_tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True
                        ) for prompt in batch_prompts
                    ]
                    
                    # 배치 입력 준비
                    inputs = self.hf_tokenizer(
                        formatted_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    ).to(self.device)
                    
                    # 생성 설정
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": 0.1,
                        "pad_token_id": self.hf_tokenizer.eos_token_id,
                        "eos_token_id": self.hf_tokenizer.eos_token_id
                    }
                    
                    # 배치 생성 (경고 메시지 제거)
                    with torch.no_grad():
                        outputs = self.hf_model.generate(
                            **inputs,
                            **generation_config
                        )
                    
                    # 배치 결과 디코딩
                    batch_results = [
                        self.hf_tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    results.extend(batch_results)
            
            return results
            
        except Exception as e:
            print(f"Error in call_llm: {str(e)}")
            return None

    def generate_message_vllm(self, messages, system_prompt, max_tokens=512):
        headers = {"Content-Type": "application/json"}
        endpoint = f"{VLLM_SERVER_URL}/chat/completions"
        
        # vLLM 서버 요구사항에 맞게 메시지 형식 수정
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        payload = {
            "model": MODEL_NAME,
            "messages": formatted_messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,  # top_k 비활성화
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "dtype": "float32"
        }
        
        for attempt in range(5):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}/5] Request failed: {e}")
                if attempt < 4:  # 마지막 시도가 아니면 잠시 대기
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("Failed to get response from vLLM server after 5 attempts")

    def parse_explanation_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        explanation_tag = soup.find("explanation")
        explanation = explanation_tag.text.strip() if explanation_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return explanation, answer

    def parse_preference_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        preference_tag = soup.find("preference")
        preference = preference_tag.text.strip() if preference_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return preference, answer
    
    def load_persona_data(self, persona_index):
        with open(PERSONA_TASK_FILE, "r", encoding="utf-8") as f:
            personas = json.load(f)
        return next(p for p in personas if p["persona_index"] == persona_index)

    def load_persona_questions(self, file_path, persona_index):
        with open(file_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        for p in personas:
            if p["persona_index"] == persona_index:
                all_qs = []
                for block in p["preference_blocks"]:
                    pref = block["preference"]
                    for q in block["queries"]:
                        all_qs.append((pref, q["question"]))
                return all_qs
        raise ValueError(f"Persona index {persona_index} not found.")

    def retrieve_top_k(self, query, index, chunks, top_k=TOP_K):
        query_emb = self.embed_query(query)
        start_retrieval = time.time()
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time

    def load_prompt_template(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    def format_prompt(self, template, preference_text, chunk_text):
        return template.replace("{preference}", preference_text).replace("{chunk}", chunk_text)

    def save_jsonl(self, file_path, items):
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def save_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_csv(self, file_path, fieldnames, row, write_header=False):
        write_header = write_header or not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def parse_decision_and_reason(self, input_string):
        """LLM 응답에서 decision과 reason 파싱"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        return decision, reason

    def process_chunk(self, chunk_text, preference_text, prompt_template):
        filled_prompt = self.format_prompt(prompt_template, preference_text, chunk_text)
        try:
            llm_response = self.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt="You are a helpful assistant for indexing document chunks."
            )
            decision, reason = self.parse_decision_and_reason(llm_response)
            if decision == "":
                return None
            return {
                "chunk": chunk_text,
                "decision": decision,
                "reason": reason
            }
        except Exception as e:
            print(f"Failed to process chunk: {e}")
            return None

    def summarize_single(self, entry, summarizing_prompt):
        original_chunk = entry["chunk"]
        reason = entry.get("reason", "")

        filled_prompt = summarizing_prompt.replace("{preference}", "N/A").replace("{chunk}", original_chunk).replace("{reason}", reason)

        summarized_text = self.generate_message_vllm(
            messages=[{"role": "user", "content": filled_prompt}],
            system_prompt="You are a helpful assistant tasked with summarizing document chunks."
        )

        return {
            "original": original_chunk,
            "summarized": summarized_text,
            "reason": reason
        }