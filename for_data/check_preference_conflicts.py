import os
import json
import requests
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

def main():
    # 입력 파일 경로
    input_file = "/data/my_PRAG/baseline/personas.json"
    output_file = "/data/my_PRAG/baseline/personas1_conflict_check.json"
    
    # persona 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # 각 persona의 선호도 상충 여부 확인
    results = []
    for persona in tqdm(personas, desc="Checking preference conflicts"):
        has_conflict, conflict_info = check_preference_conflicts(persona)
        
        result = {
            "persona_index": persona["persona_index"],
            "has_conflict": has_conflict,
            "conflict_info": conflict_info
        }
        results.append(result)
        
        # 진행 상황 출력
        if has_conflict:
            print(f"\nPersona {persona['persona_index']} has conflicting preferences:")
            print(f"Analysis: {conflict_info['analysis']}")
            print(f"Explanation: {conflict_info['explanation']}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    total_conflicts = sum(1 for r in results if r["has_conflict"])
    print(f"\n✅ Checked {len(personas)} personas")
    print(f"✅ Found {total_conflicts} personas with conflicting preferences")
    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
    main() 