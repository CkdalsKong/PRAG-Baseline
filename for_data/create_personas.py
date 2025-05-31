import os
import json
import random
import requests
from collections import defaultdict
from tqdm import tqdm

VLLM_SERVER_URL = "http://localhost:8006/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def check_preference_conflicts(persona):
    """
    LLM을 사용하여 persona의 선호도들이 서로 상충되지 않는지 확인합니다.
    
    Args:
        persona (dict): 확인할 persona 정보
    
    Returns:
        tuple: (상충 여부, 설명)
    """
    # 선호도 정보 수집
    preferences = []
    for block in persona['preference_blocks']:
        preferences.append(f"Topic: {block['topic']}\nPreference: {block['preference']}")
    
    # LLM 프롬프트 구성
    prompt = f"""Please analyze if the following preferences of a person are consistent or if there are any conflicts or contradictions between them.

List of preferences:
{chr(10).join(preferences)}

Please respond in the following format:
<analysis>
Detailed analysis of the preferences
</analysis>
<conflict>
yes or no
</conflict>
<explanation>
Explanation of whether there are conflicts and why
</explanation>
"""
    
    # LLM 호출
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(f"{VLLM_SERVER_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # 결과 파싱
        import re
        analysis = re.search(r"<analysis>(.*?)</analysis>", result, re.DOTALL)
        conflict = re.search(r"<conflict>(.*?)</conflict>", result, re.DOTALL)
        explanation = re.search(r"<explanation>(.*?)</explanation>", result, re.DOTALL)
        
        has_conflict = conflict.group(1).strip().lower() == "yes" if conflict else None
        return has_conflict, {
            "analysis": analysis.group(1).strip() if analysis else "",
            "explanation": explanation.group(1).strip() if explanation else ""
        }
    except Exception as e:
        print(f"Error checking conflicts for persona {persona['persona_index']}: {str(e)}")
        return None, {"error": str(e)}

def create_personas(input_file, output_file, num_personas=10):
    """
    Wiki 친화적인 주제들에서 각 persona가 각 주제별로 하나씩의 선호도를 가지도록 10명의 persona를 생성합니다.
    각 persona의 선호도들이 서로 상충되지 않는지 확인하고, 중복된 persona는 생성하지 않습니다.
    
    Args:
        input_file (str): wiki_friendly_preferences.json 파일 경로
        output_file (str): 생성된 persona들을 저장할 JSON 파일 경로
        num_personas (int): 생성할 persona 수
    """
    # 선호도 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        topic_preferences = json.load(f)
    
    # persona 생성
    personas = []
    used_preference_combinations = set()  # 이미 사용된 선호도 조합을 저장
    
    for i in tqdm(range(num_personas), desc="Creating personas"):
        max_attempts = 100  # 최대 시도 횟수
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            preference_blocks = []
            current_combination = []  # 현재 시도하는 선호도 조합
            
            # 각 주제별로 하나씩 선호도 선택
            for topic, prefs in topic_preferences.items():
                selected_pref = random.choice(prefs)
                preference_blocks.append({
                    'topic': topic,
                    'preference': selected_pref['preference'],
                    'queries': [
                        {
                            'question': selected_pref['question'],
                            'explanation': selected_pref['explanation']
                        }
                    ]
                })
                current_combination.append(f"{topic}:{selected_pref['preference']}")
            
            # 이미 사용된 조합인지 확인
            combination_key = "|".join(sorted(current_combination))
            if combination_key in used_preference_combinations:
                continue
            
            # persona 정보 구성
            persona = {
                'persona_index': i,
                'description': f"Persona {i+1}: A person with diverse preferences across different topics including {', '.join(topic_preferences.keys())}.",
                'preference_blocks': preference_blocks
            }
            
            # 선호도 상충 여부 확인
            has_conflict, conflict_info = check_preference_conflicts(persona)
            
            # 상충이 없거나 확인 실패한 경우에만 persona 추가
            if has_conflict is None or not has_conflict:
                if has_conflict is not None:
                    persona['conflict_check'] = {
                        'has_conflict': False,
                        'analysis': conflict_info['analysis'],
                        'explanation': conflict_info['explanation']
                    }
                personas.append(persona)
                used_preference_combinations.add(combination_key)
                print(f"\n✅ Created Persona {i+1} with consistent preferences")
                break
            else:
                print(f"\nPersona {i+1} has conflicting preferences. Regenerating...")
        
        if attempts >= max_attempts:
            print(f"\n⚠️ Could not create a unique persona {i+1} after {max_attempts} attempts")
            break
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    print(f"\n✅ Created {len(personas)} personas with {len(topic_preferences)} preferences each (one per topic)")
    print(f"✅ Saved to {output_file}")
    
    # 주제별 선호도 분포 확인
    topic_distribution = defaultdict(int)
    for persona in personas:
        for block in persona['preference_blocks']:
            topic_distribution[block['topic']] += 1
    
    print("\nTopic distribution across all personas:")
    for topic, count in topic_distribution.items():
        print(f"- {topic}: {count} preferences")

if __name__ == "__main__":
    input_file = "/data/my_PRAG/baseline/wiki_friendly_preferences.json"
    output_file = "/data/my_PRAG/baseline/personas.json"
    create_personas(input_file, output_file) 