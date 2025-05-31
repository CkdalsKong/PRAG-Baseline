import os
import json
from collections import defaultdict

def create_wiki_friendly_preferences(input_dir, output_file):
    """
    Wiki에서 잘 찾을 수 있는 주제들만 선별하여 저장합니다.
    
    Args:
        input_dir (str): explicit_preference 폴더 경로
        output_file (str): wiki 친화적인 주제만 저장할 JSON 파일 경로
    """
    topic_preferences = defaultdict(list)
    
    # Wiki 친화적인 주제 목록
    wiki_friendly_topics = {
        'education_learning_styles': '교육 및 학습 스타일',
        'education_resources': '교육 자원',
        'lifestyle_health': '건강',
        'lifestyle_dietary': '식단 및 영양',
        'travel_activities': '여행 활동',
        'travel_transportation': '교통수단',
        'entertain_sports': '스포츠',
        'entertain_music_book': '음악 및 문학',
        'shop_technology': '기술 제품'
    }
    
    # 각 JSON 파일 순회
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            topic = filename.replace('.json', '')
            
            # Wiki 친화적인 주제만 처리
            if topic in wiki_friendly_topics:
                file_path = os.path.join(input_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            topic_preferences[topic].extend(data)
                        elif isinstance(data, dict):
                            topic_preferences[topic].append(data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Wiki 친화적인 주제 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dict(topic_preferences), f, ensure_ascii=False, indent=4)
    
    print(f"✅ Wiki-friendly preferences saved to {output_file}")
    print("\nWiki-friendly topics:")
    for topic, preferences in topic_preferences.items():
        print(f"- {topic} ({wiki_friendly_topics[topic]}): {len(preferences)} preferences")

if __name__ == "__main__":
    input_dir = "/data/my_PRAG/explicit_preference"
    output_file = "/data/my_PRAG/baseline/wiki_friendly_preferences.json"
    create_wiki_friendly_preferences(input_dir, output_file) 