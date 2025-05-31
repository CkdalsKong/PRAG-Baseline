import os
import json
from tqdm import tqdm

def load_wiki_documents(file_path):
    """wiki 문서들을 로드하고 id, title, text 정보를 유지"""
    all_docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                if doc.get("text", "").strip():
                    all_docs.append(doc)  # 원본 문서 그대로 유지
            except json.JSONDecodeError:
                continue
    return all_docs

def chunk_documents(docs, chunk_size=100):
    """문서를 100단어 청크로 나누되 id와 title 정보 유지"""
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        current_chunk = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            current_chunk.extend(chunk_words)
            
            # 마지막 청크가 아니고, 현재 청크가 chunk_size에 도달했을 때
            if i + chunk_size < len(words) and len(current_chunk) >= chunk_size:
                chunk = ' '.join(current_chunk)
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": chunk
                })
                current_chunk = []
        
        # 마지막 청크 처리
        if current_chunk:
            # 마지막 청크가 20단어 미만이고 이전 청크가 있으면 합치기
            if len(current_chunk) < 20 and chunks and chunks[-1]["id"] == doc["id"]:
                chunks[-1]["text"] = chunks[-1]["text"] + " " + ' '.join(current_chunk)
            else:
                chunk = ' '.join(current_chunk)
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": chunk
                })
    
    return chunks

if __name__ == "__main__":
    input_file = "/data/my_PRAG/baseline/corpus/sampled_wiki_doc.jsonl"
    output_folder = "/data/my_PRAG/baseline/corpus"
    os.makedirs(output_folder, exist_ok=True)

    # 1. 문서 로드
    print("📂 Loading wiki documents...")
    docs = load_wiki_documents(input_file)
    print(f"✅ Loaded {len(docs)} documents")

    # 2. 청크 생성
    print("🧩 Splitting into 100-word chunks...")
    chunks = chunk_documents(docs, chunk_size=100)
    print(f"✅ Generated {len(chunks)} chunks")

    # 3. JSONL로 저장
    output_file = os.path.join(output_folder, "sampled_chunks_with_doc.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"📄 Chunks saved to: {output_file}")
    print(f"📊 Total chunks: {len(chunks)}")