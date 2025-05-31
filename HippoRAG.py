import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.nn.parallel import DataParallel

class HippoRAG:
    def __init__(self, device="cuda:0", use_multi_gpu=False):
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.model = None
        self.tokenizer = None
        self.batch_size = self._get_batch_size()
        
    def _get_batch_size(self):
        """GPU 메모리에 따른 배치 사이즈 설정"""
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            self.num_gpus = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                return 64 * self.num_gpus
            elif gpu_memory >= 48e9:  # 48GB
                return 32 * self.num_gpus
            elif gpu_memory >= 24e9:  # 24GB
                return 16 * self.num_gpus
            else:
                return 8 * self.num_gpus
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                return 128
            elif gpu_memory >= 48e9:  # 48GB
                return 64
            elif gpu_memory >= 24e9:  # 24GB
                return 32
            else:
                return 16

    def load_model(self):
        """HippoRAG 모델 로드"""
        print("Loading HippoRAG model...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever").eval()
        
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = self.model.to(self.device)
            self.model = DataParallel(
                self.model,
                device_ids=list(range(self.num_gpus)),
                output_device=0
            )
        else:
            self.model = self.model.to(self.device)
        print(f"HippoRAG model loaded on {self.device}")

    def embed_texts(self, texts):
        """텍스트 임베딩 생성"""
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                if isinstance(outputs.last_hidden_state, tuple):
                    embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(embeddings)
        return np.vstack(all_embs)

    def index(self, chunks, output_dir):
        """청크 인덱싱"""
        print("\nStarting HippoRAG indexing...")
        start_time = time.time()
        
        # 임베딩 생성
        print("Generating embeddings...")
        embeddings = self.embed_texts(chunks)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 결과 저장
        print("Saving results...")
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        self.save_jsonl(os.path.join(output_dir, "chunks.jsonl"), [{"text": chunk} for chunk in chunks])
        
        indexing_time = time.time() - start_time
        print(f"Indexing completed in {indexing_time:.2f} seconds")
        return indexing_time

    def retrieval(self, query, chunks, embeddings, top_k=5):
        """쿼리 기반 검색"""
        # 쿼리 임베딩
        query_emb = self.embed_texts([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # 유사도 계산
        similarities = np.dot(embeddings, query_emb)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [chunks[i] for i in top_indices], similarities[top_indices]

    def save_jsonl(self, file_path, items):
        """JSONL 파일 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n") 