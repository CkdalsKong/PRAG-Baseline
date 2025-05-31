import os
import json
from collections import defaultdict

def merge_topic_jsons(input_dir, output_file):
    """
    explicit_preference 폴더 내의 모든 JSON 파일들을 주제별로 묶어서 저장합니다.
    
    Args:
        input_dir (str): explicit_preference 폴더 경로
        output_file (str): 출력할 JSON 파일 경로
    """
    topic_preferences = defaultdict(list)
    
    # 각 JSON 파일 순회
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            topic = filename.replace('.json', '')  # 주제 이름 추출
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        topic_preferences[topic].extend(data)
                    elif isinstance(data, dict):
                        topic_preferences[topic].append(data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # 결과를 JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dict(topic_preferences), f, ensure_ascii=False, indent=4)
    
    print(f"✅ Merged preferences saved to {output_file}")
    print("\nTopic statistics:")
    for topic, preferences in topic_preferences.items():
        print(f"- {topic}: {len(preferences)} preferences")

if __name__ == "__main__":
    input_dir = "/data/my_PRAG/explicit_preference"
    output_file = "/data/my_PRAG/baseline/explicit_preference.json"
    merge_topic_jsons(input_dir, output_file) 