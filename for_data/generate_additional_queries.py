import os
import json
import requests
from tqdm import tqdm

VLLM_SERVER_URL = "http://localhost:8006/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def generate_additional_queries(preference, example_question, example_explanation):
    """
    주어진 선호도와 예시 질문을 바탕으로 추가 질문을 생성합니다.
    
    Args:
        preference (str): 선호도 텍스트
        example_question (str): 예시 질문
        example_explanation (str): 예시 설명
    
    Returns:
        list: 생성된 추가 질문과 설명 리스트
    """
    prompt = f"""Given a preference and an example question, generate 2 additional questions that could be asked about this preference.
The questions should be Wikipedia-friendly, meaning they should be:
1. Factual and objective
2. Suitable for finding information in Wikipedia articles
3. Focused on general knowledge rather than personal opinions
4. Clear and specific enough to be answered with Wikipedia content

Preference: {preference}

Example question: {example_question}
Example explanation: {example_explanation}

Please generate 2 additional Wikipedia-friendly questions and their explanations in the following format:
<question1>
First additional question
</question1>
<explanation1>
Explanation for the first question
</explanation1>
<question2>
Second additional question
</question2>
<explanation2>
Explanation for the second question
</explanation2>
"""
    
    # LLM 호출
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(f"{VLLM_SERVER_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # 결과 파싱
        import re
        question1 = re.search(r"<question1>(.*?)</question1>", result, re.DOTALL)
        explanation1 = re.search(r"<explanation1>(.*?)</explanation1>", result, re.DOTALL)
        question2 = re.search(r"<question2>(.*?)</question2>", result, re.DOTALL)
        explanation2 = re.search(r"<explanation2>(.*?)</explanation2>", result, re.DOTALL)
        
        additional_queries = []
        if question1 and explanation1:
            additional_queries.append({
                "question": question1.group(1).strip(),
                "explanation": explanation1.group(1).strip()
            })
        if question2 and explanation2:
            additional_queries.append({
                "question": question2.group(1).strip(),
                "explanation": explanation2.group(1).strip()
            })
        
        return additional_queries
    except Exception as e:
        print(f"Error generating additional queries: {str(e)}")
        return []

def main():
    # 입력 파일 경로
    input_file = "/data/my_PRAG/baseline/personas.json"
    output_file = "/data/my_PRAG/baseline/personas_with_additional_queries.json"
    
    # persona 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # 각 persona의 각 토픽에 대해 추가 질문 생성
    for persona in tqdm(personas, desc="Generating additional queries"):
        for block in persona['preference_blocks']:
            # 기존 질문을 예시로 사용
            example_query = block['queries'][0]
            
            # 추가 질문 생성
            additional_queries = generate_additional_queries(
                block['preference'],
                example_query['question'],
                example_query['explanation']
            )
            
            # 기존 질문과 추가 질문 합치기
            block['queries'].extend(additional_queries)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    total_queries = sum(len(block['queries']) for persona in personas for block in persona['preference_blocks'])
    print(f"\n✅ Added additional queries to {len(personas)} personas")
    print(f"✅ Total queries: {total_queries}")
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    main() 