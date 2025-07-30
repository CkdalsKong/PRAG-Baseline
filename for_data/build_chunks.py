import os
import re
import json
import random
from tqdm import tqdm
import argparse

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

def load_chunks(file_path):
    """이미 생성된 청크들을 로드"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunk = json.loads(line)
                if chunk.get("text", "").strip():
                    chunks.append(chunk)
            except json.JSONDecodeError:
                continue
    return chunks

def clean_text(text):
    # &lt;...&gt; 형태의 태그 제거
    text = re.sub(r'&lt;.*?&gt;', '', text)
    return text.strip()

def chunk_documents_sentencewise(docs, chunk_size=100):
    """문서들을 문장 단위로 쪼개고, 100단어 이하로 묶어서 chunk 생성 (불필요한 태그 제거 포함)"""
    chunks = []
    for doc in docs:
        # 1. 불필요한 태그 제거
        clean_doc_text = clean_text(doc["text"])
        # 2. 문장 단위로 쪼개기 (마침표, 느낌표, 물음표, 줄바꿈 등)
        sentences = re.split(r'(?<=[.!?]) +', clean_doc_text)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            n_words = len(words)
            if n_words == 0:
                continue

            # 만약 한 문장이 chunk_size보다 길면, 단독 chunk로
            if n_words > chunk_size:
                if current_chunk:
                    chunks.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": " ".join(current_chunk)
                    })
                    current_chunk = []
                    current_length = 0
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": sentence
                })
                continue
            # chunk에 문장을 추가했을 때 chunk_size를 넘으면, 새 chunk 시작
            if current_length + n_words > chunk_size:
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": " ".join(current_chunk)
                })
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += n_words

        # 마지막 chunk 처리
        if current_chunk:
            chunks.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": " ".join(current_chunk)
            })

    return chunks
    
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

def sample_chunks_for_baseline(chunks, sample_ratio=0.01):
    """기존 청크들에서 랜덤 샘플링"""
    random.seed(42)  # 고정된 시드 사용
    sampled_chunks = []
    
    for chunk in chunks:
        if random.random() < sample_ratio:
            sampled_chunks.append(chunk)
    
    return sampled_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build chunks from wiki documents")
    parser.add_argument("--doc_mode", type=str, required=True, 
                        choices=["total", "sample", "related", "related2", "sample_sw", "total_sw", "baseline_sampling"],
                        help="Document mode: 'total' for all documents, 'sample' for sampled documents, 'related' for sampled related documents, 'baseline_sampling' for sampling from existing chunks")
    parser.add_argument("--sample_ratio", type=float, default=0.01,
                        help="Sampling ratio for baseline_sampling mode (default: 0.01 = 1%)")
    args = parser.parse_args()

    if args.doc_mode == "baseline_sampling":
        # baseline_sampling 모드: 기존 청크에서 샘플링
        input_file = "../data/corpus/full_chunks_with_doc_sw.jsonl"
        
        # 샘플링 비율에 따라 파일명 결정
        if args.sample_ratio == 0.001:  # 0.1%
            output_file = "../data/corpus/forbaseline_sampled_chunks_with_doc_sw_01.jsonl"
        else:  # 1% (기본값)
            output_file = "../data/corpus/forbaseline_sampled_chunks_with_doc_sw.jsonl"
        
        print(f"📂 Loading existing chunks from {input_file}...")
        chunks = load_chunks(input_file)
        print(f"✅ Loaded {len(chunks)} chunks")
        
        print(f"🎲 Sampling {args.sample_ratio*100}% of chunks (seed=42)...")
        sampled_chunks = sample_chunks_for_baseline(chunks, args.sample_ratio)
        print(f"✅ Sampled {len(sampled_chunks)} chunks")
        
        # JSONL로 저장
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in sampled_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"📄 Sampled chunks saved to: {output_file}")
        print(f"📊 Original chunks: {len(chunks)}, Sampled chunks: {len(sampled_chunks)}")
        
    else:
        # 기존 모드들
        if args.doc_mode in ["total", "total_sw"]:
            input_file = "../data/corpus/full_wiki_doc.jsonl"
        elif args.doc_mode == "related":
            input_file = "../data/corpus/sampled_related_wiki_doc.jsonl"
        elif args.doc_mode == "related2":
            input_file = "../data/corpus/sampled_related_wiki_doc2.jsonl"
        else:
            input_file = "../data/corpus/sampled_wiki_doc.jsonl"
        output_folder = "../data/corpus"
        os.makedirs(output_folder, exist_ok=True)

        # 1. 문서 로드
        print("📂 Loading wiki documents...")
        docs = load_wiki_documents(input_file)
        print(f"✅ Loaded {len(docs)} documents")

        # 2. 청크 생성
        if args.doc_mode in ["sample_sw", "total_sw"]:
            print("🧩 Splitting into 100-word sentencewise chunks...")
            chunks = chunk_documents_sentencewise(docs, chunk_size=100)
        else:
            print("🧩 Splitting into 100-word chunks...")
            chunks = chunk_documents(docs, chunk_size=100)
        print(f"✅ Generated {len(chunks)} chunks")

        # 3. JSONL로 저장
        if args.doc_mode == "total":
            output_file = os.path.join(output_folder, "full_chunks_with_doc.jsonl")
        elif args.doc_mode == "related":
            output_file = os.path.join(output_folder, "sampled_related_chunks_with_doc.jsonl")
        elif args.doc_mode == "related2":
            output_file = os.path.join(output_folder, "sampled_related_chunks_with_doc2.jsonl")
        elif args.doc_mode == "sample_sw":
            output_file = os.path.join(output_folder, "sampled_chunks_with_doc_sw.jsonl")
        elif args.doc_mode == "total_sw":
            output_file = os.path.join(output_folder, "full_chunks_with_doc_sw.jsonl")
        else:
            output_file = os.path.join(output_folder, "sampled_chunks_with_doc.jsonl")
            
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"📄 Chunks saved to: {output_file}")
        print(f"📊 Total chunks: {len(chunks)}")