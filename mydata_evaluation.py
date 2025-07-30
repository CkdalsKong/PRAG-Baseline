import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydata_utils import MyDataUtils, EVALUATION_REPORT_FILE, ERROR_TYPE_DIR

class MyDataEvaluation:
    def __init__(self, utils):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.use_multi_gpu = utils.use_multi_gpu
        self.chunk_mode = utils.chunk_mode
        self.output_dir = utils.output_dir
        self.persona_task_file = utils.persona_task_file
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.chunk_file = utils.chunk_file
        self.embedding_file = utils.embedding_file
        self.batch_size = utils.batch_size
        self.use_trust_align = utils.use_trust_align
        
        # 임베딩 파일 경로 설정
        self.embeddings_file = os.path.join(self.output_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        self.index_file = os.path.join(self.output_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

    def process_metric(self, task, metric, eval_message_text, system_prompt):
        preference = task["preference"]
        question = task["question"]
        end_generation = task["response_to_q"]
        error_check = task.get("evaluation_error_analysis", {})
        
        if metric in error_check:
            return None
        
        eval_text = eval_message_text
        if metric == "acknow":
            eval_text = eval_text.replace("{end_generation}", end_generation).replace("{question}", question)
        elif metric in ["violate", "helpful"]:
            eval_text = eval_text.replace("{preference}", preference).replace("{question}", question).replace("{end_generation}", end_generation)
        elif metric == "hallucinate":
            extracted_pref = error_check.get("acknow", {}).get("extract_pref", "")
            eval_text = eval_text.replace("{preference}", preference).replace("{assistant_restatement}", extracted_pref)
        
        eval_message = [{"role": "user", "content": eval_text}]
        try:
            eval_response = self.utils.generate_message_vllm(eval_message, system_prompt)
            
            result = {}
            if metric != "acknow":
                explanation, answer = self.utils.parse_explanation_and_answer(eval_response)
                result["explanation"] = explanation
                result["answer"] = answer
            else:
                extract_preference, answer = self.utils.parse_preference_and_answer(eval_response)
                result["answer"] = answer
                result["extract_pref"] = extract_preference
            
            return metric, result
        except Exception as e:
            print(f"⚠️ Evaluation call failed: {str(e)}")
            return None
    
    def run_evaluation_with_cache(self, persona_index, method_dir, cached_resources):
        """캐시된 리소스를 사용하는 새로운 evaluation 방식"""
        # 평가 메트릭 로드
        file_dict = {
            "acknow": "check_acknowledge.txt",
            "violate": "check_violation.txt",
            "hallucinate": "check_hallucination.txt",
            "helpful": "check_helpful.txt"
        }
        
        eval_message_texts = []
        for metric_name, file_name in file_dict.items():
            file_path = os.path.join(ERROR_TYPE_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                eval_message_texts.append((metric_name, f.read()))

        system_prompt = "You are a helpful assistant in evaluating an AI assistant's response. You should be fair and strict and follow the user's instruction."
        
        # Generation 결과 로드
        if self.utils.index_type == "flat":
            if self.method[-2:] == "wq":
                generation_file = os.path.join(method_dir, f"gen_wq_flat_{persona_index}.json")
            elif self.method[-3:] == "wql":
                generation_file = os.path.join(method_dir, f"gen_wql_flat_{persona_index}.json")
            else:
                generation_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")
        elif self.method == "raptor":
            generation_file = os.path.join(method_dir, f"gen_raptor_{persona_index}.json")
        elif self.use_trust_align:
            generation_file = os.path.join(method_dir, f"gen_{self.method}_trustalign_{persona_index}.json")
        else:
            if self.method[-2:] == "wq":
                generation_file = os.path.join(method_dir, f"gen_wq_{persona_index}.json")
            elif self.method[-3:] == "wql":
                generation_file = os.path.join(method_dir, f"gen_wql_{persona_index}.json")
            else:
                generation_file = os.path.join(method_dir, f"gen_{self.method}_{persona_index}.json")
        generation_data = self.utils.load_json(generation_file)
        
        print(f"✅ Loaded {len(generation_data)} generations from {generation_file}")
        
        # 평가 결과 저장 파일
        if self.utils.index_type == "flat":
            if self.method[-2:] == "wq":
                save_file = os.path.join(method_dir, f"eval_wq_flat_{persona_index}.json")
            elif self.method[-3:] == "wql":
                save_file = os.path.join(method_dir, f"eval_wql_flat_{persona_index}.json")
            else:
                save_file = os.path.join(method_dir, f"eval_{self.method}_flat_{persona_index}.json")
        elif self.method == "raptor":
            save_file = os.path.join(method_dir, f"eval_raptor_{persona_index}.json")
        elif self.use_trust_align:
            save_file = os.path.join(method_dir, f"eval_{self.method}_trustalign_{persona_index}.json")
        else:
            if self.method[-2:] == "wq":
                save_file = os.path.join(method_dir, f"eval_wq_{persona_index}.json")
            elif self.method[-3:] == "wql":
                save_file = os.path.join(method_dir, f"eval_wql_{persona_index}.json")
            else:
                save_file = os.path.join(method_dir, f"eval_{self.method}_{persona_index}.json")
        
        # 캐시된 모델 사용 (모델 로딩 생략)
        print("✅ Using cached models for evaluation")

        # 평가 실행
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for task_id, task in enumerate(generation_data):
                if "response_to_q" not in task:
                    print(f"⚠️ Skipped (no response) - Task ID: {task_id}")
                    continue
                
                if "evaluation_error_analysis" in task:
                    analysis = task["evaluation_error_analysis"]
                    if all(k in analysis for k in ["acknow", "violate", "hallucinate", "helpful"]):
                        continue
                
                # 각 메트릭에 대해 future 생성
                for metric, eval_message_text in eval_message_texts:
                    future = executor.submit(
                        self.process_metric,
                        task,
                        metric,
                        eval_message_text,
                        system_prompt
                    )
                    futures.append((task_id, future))
            
            # 결과 수집
            for task_id, future in tqdm(futures, desc="Evaluating responses"):
                result = future.result()
                if result:
                    metric, error_check = result
                    if "evaluation_error_analysis" not in generation_data[task_id]:
                        generation_data[task_id]["evaluation_error_analysis"] = {}
                    generation_data[task_id]["evaluation_error_analysis"][metric] = error_check
                    
                    # 중간 결과 저장
                    self.utils.save_json(save_file, generation_data)

        print(f"✅ Evaluation for persona {persona_index} finished. Saved to {save_file}")

        # 평가 결과 분석
        stats = {
            "acknowledgement": 0,
            "hallucination": 0,
            "violation": 0,
            "error_unhelpful": 0,
            "error_inconsistent": 0,
            "hallucination_of_preference_violation": 0,
            "preference_unaware_violation": 0,
            "preference_adherence_accuracy": 0,
        }
        
        for idx, entry in tqdm(enumerate(generation_data), total=len(generation_data), desc="Analyzing Errors"):
            if "evaluation_error_analysis" not in entry:
                print(f"⚠️ Warning: Entry {idx} has not been evaluated yet!")
                continue
            
            error_types = entry["evaluation_error_analysis"]
            is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
            is_hallucination = is_acknowledgement and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
            is_violation = "yes" in error_types.get("violate", {}).get("answer", "").lower()
            is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()
            
            is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
            is_hallucination_of_preference_violation = (
                is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
            )
            is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful
            
            preference_following_accuracy = not any(
                [is_inconsistent, is_hallucination_of_preference_violation, is_preference_unaware_violation, is_unhelpful]
            )
            
            stats["acknowledgement"] += is_acknowledgement
            stats["hallucination"] += is_hallucination
            stats["violation"] += is_violation
            stats["error_unhelpful"] += is_unhelpful
            stats["error_inconsistent"] += is_inconsistent
            stats["hallucination_of_preference_violation"] += is_hallucination_of_preference_violation
            stats["preference_unaware_violation"] += is_preference_unaware_violation
            stats["preference_adherence_accuracy"] += preference_following_accuracy
        
        # 분석 결과 저장
        fieldnames = [
            "method",
            "persona_index",
            "unhelpful",
            "inconsistent",
            "hallucination_of_preference_violation",
            "preference_unaware_violation",
            "preference_following_accuracy(%)"
        ]
        
        row = {
            "method": self.method,
            "persona_index": persona_index,
            "unhelpful": stats["error_unhelpful"],
            "inconsistent": stats["error_inconsistent"],
            "hallucination_of_preference_violation": stats["hallucination_of_preference_violation"],
            "preference_unaware_violation": stats["preference_unaware_violation"],
            "preference_following_accuracy(%)": round((stats["preference_adherence_accuracy"] / len(generation_data)) * 100, 2)
        }
        
        self.utils.save_csv(os.path.join(self.output_dir, EVALUATION_REPORT_FILE), fieldnames, row)
        
        print(f"✅ Evaluation summary saved to: {os.path.join(self.output_dir, EVALUATION_REPORT_FILE)}")
        return method_dir