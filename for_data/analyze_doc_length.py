import os
import json
from tqdm import tqdm
import numpy as np

def analyze_doc_lengths(directory):
    """
    주어진 디렉토리의 모든 JSONL 파일에서 문서 길이를 분석합니다.
    
    Args:
        directory (str): JSONL 파일들이 있는 디렉토리 경로
    """
    all_lengths = []
    
    # 모든 JSONL 파일 처리
    for filename in tqdm(os.listdir(directory), desc="Processing files"):
        if not filename.endswith('.jsonl'):
            continue
            
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 각 줄을 JSON으로 파싱
            for line in f:
                doc = json.loads(line)
                if isinstance(doc, dict) and "text" in doc:
                    doc_length = len(doc["text"].split())
                    all_lengths.append(doc_length)
    
    # 통계 계산
    if all_lengths:
        lengths = np.array(all_lengths)
        print(f"\n문서 길이 통계:")
        print(f"총 문서 수: {len(lengths):,}")
        print(f"최소 길이: {lengths.min():,} 단어")
        print(f"최대 길이: {lengths.max():,} 단어")
        print(f"평균 길이: {lengths.mean():,.2f} 단어")
        print(f"중앙값 길이: {np.median(lengths):,.2f} 단어")
        print(f"표준편차: {lengths.std():,.2f} 단어")
        
        # 길이 분포 출력
        percentiles = [25, 50, 75, 90, 95, 99]
        print("\n길이 분포 (백분위):")
        for p in percentiles:
            print(f"{p}%: {np.percentile(lengths, p):,.2f} 단어")
    else:
        print("분석할 문서를 찾을 수 없습니다.")

if __name__ == "__main__":
    directory = "/data/my_PRAG/filtered_wiki_json"
    analyze_doc_lengths(directory) 