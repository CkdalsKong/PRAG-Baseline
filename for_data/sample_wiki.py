import os
import json
import random
from tqdm import tqdm

# 입력/출력 디렉토리 설정
INPUT_DIR = "/data/my_PRAG/filtered_wiki_json"
OUTPUT_DIR = "/data/my_PRAG/baseline/corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 샘플링 비율 설정
SAMPLE_RATIO = 0.05

def sample_wiki_documents():
    # 모든 jsonl 파일 목록 가져오기
    jsonl_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl')]
    print(f"Found {len(jsonl_files)} jsonl files")
    
    # 전체 문서 수 카운트
    total_docs = 0
    for file in tqdm(jsonl_files, desc="Counting total documents"):
        with open(os.path.join(INPUT_DIR, file), 'r', encoding='utf-8') as f:
            total_docs += sum(1 for _ in f)
    
    # 샘플링할 문서 수 계산
    sample_size = int(total_docs * SAMPLE_RATIO)
    print(f"Total documents: {total_docs}")
    print(f"Will sample {sample_size} documents ({SAMPLE_RATIO*100}%)")
    
    # 랜덤 시드 설정
    random.seed(42)
    
    # 샘플링된 문서 저장
    sampled_docs = []
    doc_id = 0
    
    # 각 파일에서 문서 샘플링
    for file in tqdm(jsonl_files, desc="Sampling documents"):
        with open(os.path.join(INPUT_DIR, file), 'r', encoding='utf-8') as f:
            for line in f:
                if random.random() < SAMPLE_RATIO:
                    doc = json.loads(line)
                    sampled_docs.append({
                        "id": doc_id,
                        "title": doc["title"],
                        "text": doc["text"]
                    })
                    doc_id += 1
    
    # 샘플링된 문서 저장
    output_file = os.path.join(OUTPUT_DIR, "sampled_wiki_doc.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in sampled_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Sampled {len(sampled_docs)} documents")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    sample_wiki_documents() 