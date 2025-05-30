import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.nn.parallel import DataParallel

# 설정
ROOT_DIR = "/data/my_PRAG/baseline"
CHUNK_FILE = os.path.join(ROOT_DIR, "corpus/sampled_chunks_with_doc.jsonl")
EMBEDDING_FILE = os.path.join(ROOT_DIR, "corpus/sampled_embeddings_with_doc.npy")
DEVICE = "cuda:0"
BATCH_SIZE = 256

def load_chunks(file_path):
    """JSONL 파일에서 청크 로드"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
    return chunks

def load_models():
    """Contriever 모델과 토크나이저 로드"""
    print("Loading Contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").eval()
    
    # GPU 메모리 관리를 위한 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = model.to(DEVICE)
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = model.to(DEVICE)
    
    return tokenizer, model

def generate_embeddings(chunks, tokenizer, model):
    """청크들의 임베딩 생성"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Generating embeddings"):
        batch = chunks[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            if isinstance(outputs.last_hidden_state, tuple):
                embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
            else:
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

def main():
    # 1. 청크 로드
    print("Loading chunks...")
    chunks = load_chunks(CHUNK_FILE)
    print(f"Loaded {len(chunks)} chunks")
    
    # 2. 모델 로드
    tokenizer, model = load_models()
    
    # 3. 임베딩 생성
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, tokenizer, model)
    
    # 4. 임베딩 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 5. 결과 저장
    print(f"Saving {len(embeddings)} embeddings to {EMBEDDING_FILE}")
    np.save(EMBEDDING_FILE, embeddings)
    print("Done!")

if __name__ == "__main__":
    main() 