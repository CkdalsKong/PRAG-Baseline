import os
import json
import re
from tqdm import tqdm

def parse_answers(prompts_file, personas_file, output_file):
    """
    LLM의 답변을 파싱하여 원래 persona의 queries에 추가합니다.
    
    Args:
        prompts_file (str): prompts_for_additional_queries.json 파일 경로
        personas_file (str): personas.json 파일 경로
        output_file (str): 수정된 persona를 저장할 JSON 파일 경로
    """
    # 프롬프트와 답변 데이터 로드
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    # persona 데이터 로드
    with open(personas_file, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # 각 프롬프트의 답변 파싱하여 해당 persona에 추가
    for prompt in tqdm(prompts, desc="Parsing answers"):
        if not prompt['answer']:
            continue
            
        # 해당하는 persona 찾기
        persona = next((p for p in personas if p['persona_index'] == prompt['persona_index']), None)
        if not persona:
            continue
            
        # 해당하는 preference block 찾기
        block = next((b for b in persona['preference_blocks'] if b['topic'] == prompt['topic']), None)
        if not block:
            continue
            
        # 답변에서 질문과 설명 추출
        question1 = re.search(r"<question1>(.*?)</question1>", prompt['answer'], re.DOTALL)
        explanation1 = re.search(r"<explanation1>(.*?)</explanation1>", prompt['answer'], re.DOTALL)
        question2 = re.search(r"<question2>(.*?)</question2>", prompt['answer'], re.DOTALL)
        explanation2 = re.search(r"<explanation2>(.*?)</explanation2>", prompt['answer'], re.DOTALL)
        
        # 파싱된 결과를 queries에 추가
        if question1 and explanation1:
            block['queries'].append({
                'question': question1.group(1).strip(),
                'explanation': explanation1.group(1).strip()
            })
            
        if question2 and explanation2:
            block['queries'].append({
                'question': question2.group(1).strip(),
                'explanation': explanation2.group(1).strip()
            })
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    total_queries = sum(len(block['queries']) for persona in personas for block in persona['preference_blocks'])
    print(f"\n✅ Added additional queries to personas")
    print(f"✅ Total queries: {total_queries}")
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    prompts_file = "/data/my_PRAG/baseline/prompts_for_additional_queries.json"
    personas_file = "/data/my_PRAG/baseline/personas.json"
    output_file = "/data/my_PRAG/baseline/personas_with_additional_queries.json"
    parse_answers(prompts_file, personas_file, output_file) 