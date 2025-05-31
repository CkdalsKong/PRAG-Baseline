import os
import json
from tqdm import tqdm
import re

def split_into_chunks(text, chunk_size=100, overlap=10):
    """
    텍스트를 청크로 분할합니다.
    
    Args:
        text (str): 분할할 텍스트
        chunk_size (int): 각 청크의 최대 단어 수
        overlap (int): 청크 간 중복 단어 수
    
    Returns:
        list: 청크 리스트
    """
    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        # 현재 청크에 문장을 추가했을 때 chunk_size를 초과하는 경우
        if current_length + sentence_length > chunk_size:
            if current_chunk:  # 현재 청크가 비어있지 않은 경우
                chunks.append(' '.join(current_chunk))
                # overlap만큼의 단어를 다음 청크의 시작으로 사용
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words
                current_length = len(overlap_words)
        
        # 문장이 chunk_size보다 큰 경우
        if sentence_length > chunk_size:
            # 현재 청크가 있으면 저장
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # 긴 문장을 chunk_size 단위로 분할
            for i in range(0, sentence_length, chunk_size - overlap):
                chunk = sentence_words[i:i + chunk_size]
                chunks.append(' '.join(chunk))
        else:
            current_chunk.extend(sentence_words)
            current_length += sentence_length
    
    # 마지막 청크 처리
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_documents(input_dir, output_dir, min_words=50):
    """
    JSONL 파일의 문서들을 청크로 분할합니다.
    
    Args:
        input_dir (str): 입력 JSONL 파일 디렉토리
        output_dir (str): 출력 JSONL 파일 디렉토리
        min_words (int): 처리할 최소 단어 수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 통계를 위한 변수
    total_docs = 0
    filtered_docs = 0
    total_chunks = 0
    
    # 모든 JSONL 파일 처리
    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        if not filename.endswith('.jsonl'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"chunked_{filename}")
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                total_docs += 1
                doc = json.loads(line)
                if isinstance(doc, dict) and "text" in doc:
                    # 문서 길이 확인
                    doc_length = len(doc["text"].split())
                    
                    # 길이 필터링
                    if doc_length < min_words:
                        continue
                        
                    filtered_docs += 1
                    # 문서를 청크로 분할
                    chunks = split_into_chunks(doc["text"])
                    total_chunks += len(chunks)
                    
                    # 각 청크를 새로운 문서로 저장
                    for i, chunk in enumerate(chunks):
                        chunk_doc = {
                            "id": f"{doc['id']}_{i}",
                            "title": doc["title"],
                            "text": chunk,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_length": doc_length
                        }
                        f_out.write(json.dumps(chunk_doc, ensure_ascii=False) + '\n')
    
    # 통계 출력
    print(f"\n처리 결과:")
    print(f"총 문서 수: {total_docs:,}")
    print(f"필터링 후 문서 수: {filtered_docs:,} ({filtered_docs/total_docs*100:.1f}%)")
    print(f"생성된 청크 수: {total_chunks:,}")
    print(f"문서당 평균 청크 수: {total_chunks/filtered_docs:.1f}")

if __name__ == "__main__":
    input_dir = "/data/my_PRAG/filtered_wiki_json"
    output_dir = "/data/my_PRAG/baseline/chunked_corpus"
    process_documents(input_dir, output_dir, min_words=50) 