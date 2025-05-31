import os
import json
from tqdm import tqdm

def generate_prompts(input_file, output_file):
    """
    각 persona의 preference와 질문, 설명을 포함한 프롬프트를 생성합니다.
    
    Args:
        input_file (str): personas.json 파일 경로
        output_file (str): 생성된 프롬프트를 저장할 JSON 파일 경로
    """
    # persona 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    prompts = []
    for persona in tqdm(personas, desc="Generating prompts"):
        for block in persona['preference_blocks']:
            prompt = f"""Given a preference and an example question, generate 2 additional questions that could be asked about this preference.
The questions should be designed to test how well a system can understand and respect the user's preferences when providing information from Wikipedia articles.

Key requirements for the questions:
1. Questions should be Wikipedia-friendly (factual, objective, and answerable using Wikipedia content)
2. Questions should have a high probability of preference violation (P(answer|question) >> P(answer|preference, question))
3. Questions should not directly contradict the preference
4. Questions should be specific enough to require the system to consider the preference when searching and selecting Wikipedia content
5. Questions should be realistic and practical for a user with this preference to ask

Preference: {block['preference']}

Example question: {block['queries'][0]['question']}
Example explanation: {block['queries'][0]['explanation']}

Please generate 2 additional questions and their explanations in the following format:
<question1>
First additional question
</question1>
<explanation1>
Explanation of how this question tests the system's ability to respect the user's preference while using Wikipedia content, and why it has a high probability of preference violation
</explanation1>
<question2>
Second additional question
</question2>
<explanation2>
Explanation of how this question tests the system's ability to respect the user's preference while using Wikipedia content, and why it has a high probability of preference violation
</explanation2>
"""
            prompts.append({
                'persona_index': persona['persona_index'],
                'topic': block['topic'],
                'preference': block['preference'],
                'example_question': block['queries'][0]['question'],
                'example_explanation': block['queries'][0]['explanation'],
                'prompt': prompt,
                'answer': ''  # LLM의 답변을 저장할 필드
            })
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    print(f"\n✅ Generated {len(prompts)} prompts")
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    input_file = "/data/my_PRAG/baseline/personas.json"
    output_file = "/data/my_PRAG/baseline/prompts_for_additional_queries.json"
    generate_prompts(input_file, output_file) 