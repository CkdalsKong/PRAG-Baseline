import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.nn.parallel import DataParallel
import argparse

def load_chunks(file_path):
    """JSONL 파일에서 청크 로드"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
    return chunks

def load_models(model_name):
    """임베딩 모델과 토크나이저 로드"""
    print(f"Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    
    # GPU 메모리 관리를 위한 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = model.to(device)
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = model.to(device)
    
    return tokenizer, model, device

def generate_embeddings(chunks, tokenizer, model, device, batch_size=256):
    """청크들의 임베딩 생성"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            if isinstance(outputs.last_hidden_state, tuple):
                embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
            else:
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/contriever", help="임베딩 모델 이름")
    parser.add_argument("--chunk_file", type=str, required=True, help="청크 파일 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 디렉토리 경로")
    parser.add_argument("--batch_size", type=int, default=256, help="배치 사이즈")
    args = parser.parse_args()

    # 모델 이름에서 파일명 생성
    model_name_clean = args.model_name.replace("/", "_")
    
    # 1. 청크 로드
    print("Loading chunks...")
    chunks = load_chunks(args.chunk_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # 2. 모델 로드
    tokenizer, model, device = load_models(args.model_name)
    
    # 3. 임베딩 생성
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, tokenizer, model, device, args.batch_size)
    
    # 4. 임베딩 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 5. 결과 저장
    output_file = os.path.join(args.output_dir, f"embeddings_{model_name_clean}.npy")
    print(f"Saving {len(embeddings)} embeddings to {output_file}")
    np.save(output_file, embeddings)
    print("Done!")

if __name__ == "__main__":
    main() 