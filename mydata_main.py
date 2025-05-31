import os
import json
import time
import argparse
from mydata_utils import MyDataUtils
from mydata_indexing import MyDataIndexing
from mydata_generation import MyDataGeneration
from mydata_evaluation import MyDataEvaluation

class MyDataMain:
    def __init__(self, mode="all", method="all", device="cuda:0", use_multi_gpu=False, chunk_mode="wodoc", output_dir=None, persona_task_file=None, emb_model_name="facebook/contriever"):
        self.mode = mode
        self.method = method
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.chunk_mode = chunk_mode
        self.output_dir = output_dir
        self.persona_task_file = persona_task_file
        self.emb_model_name = emb_model_name
        
        # 유틸리티 클래스 초기화
        self.utils = MyDataUtils(
            mode=mode,
            method=method,
            device=device,
            use_multi_gpu=use_multi_gpu,
            chunk_mode=chunk_mode,
            output_dir=output_dir,
            persona_task_file=persona_task_file,
            emb_model_name=emb_model_name
        )
        
        # 각 단계별 처리 클래스 초기화
        self.indexing = MyDataIndexing(self.utils)
        self.generation = MyDataGeneration(self.utils)
        self.evaluation = MyDataEvaluation(self.utils)
    
    def run(self, persona_index):
        if persona_index == -1:
            print(f"\n=== Starting MyData pipeline with method {self.method} ===")
        else:
            print(f"\n=== Starting MyData pipeline for persona {persona_index} with method {self.method} ===")
        
        # 출력 디렉토리 설정
        if self.method in ["standard", "random", "hipporag"]:
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}")
        else:
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}/{persona_index}")
        os.makedirs(method_dir, exist_ok=True)
        
        # 1. Indexing
        if self.mode in ["indexing", "all"]:
            print("\n1. Starting indexing...")
            faiss_index_path = os.path.join(method_dir, "faiss.index")
            if os.path.exists(faiss_index_path):
                print(f"✅ Indexing already completed. Skipping...")
            else:
                self.indexing.run_indexing(persona_index)
                print(f"✅ Indexing completed. Results saved to {method_dir}")
        
        # 2. Generation
        if self.mode in ["generation", "all"] and persona_index != -1:
            print("\n2. Starting generation...")
            gen_file = os.path.join(method_dir, f"gen_{self.method}_{persona_index}.json")
            if os.path.exists(gen_file):
                print(f"✅ Generation already completed. Skipping...")
            else:
                self.generation.run_generation(persona_index, method_dir)
                print(f"✅ Generation completed. Results saved to {gen_file}")
        
        # 3. Evaluation
        if self.mode in ["evaluation", "all"] and persona_index != -1:
            print("\n3. Starting evaluation...")
            eval_file = os.path.join(method_dir, f"eval_{self.method}_{persona_index}.json")
            if os.path.exists(eval_file):
                print(f"✅ Evaluation already completed. Skipping...")
            else:
                self.evaluation.run_evaluation(persona_index, method_dir)
                print(f"✅ Evaluation completed. Results saved to {eval_file}")
        
        if persona_index == -1:
            print(f"\n=== Completed MyData pipeline ===")
        else:
            print(f"\n=== Completed MyData pipeline for persona {persona_index} ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["naive_p", "standard", "cosine_only", "random", "hipporag", "all"], help="Method type: 'naive_p', 'standard', 'cosine_only', or 'all'")
    parser.add_argument("--persona_index", type=str, required=True, help="Persona index (0-10) or 'all'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--mode", type=str, required=True, choices=["indexing", "generation", "evaluation", "all"], help="Mode to run: 'indexing', 'generation', 'evaluation', or 'all'")
    parser.add_argument("--use_multi_gpu", action="store_true", help="멀티 GPU 사용 여부")
    parser.add_argument("--chunk_mode", type=str, required=True, choices=["wodoc", "wdoc"], help="Chunk mode: 'wodoc' for chunks without document info, 'wdoc' for chunks with document info")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--persona_task_file", type=str, required=True, help="Path to the persona task file")
    parser.add_argument("--emb_model_name", type=str, default="facebook/contriever", help="임베딩 모델 이름")
    args = parser.parse_args()
    
    # Persona 인덱스 설정
    if args.persona_task_file == "final_persona_tasks.json":
        indices = list(range(10)) if args.persona_index == "all" else [int(args.persona_index)]
    else:
        indices = list(range(20)) if args.persona_index == "all" else [int(args.persona_index)]
    
    # 실행할 방법들 설정
    methods = ["naive_p", "standard", "cosine_only", "random", "hipporag"] if args.method == "all" else [args.method]
    
    # 각 방법에 대해 실행
    for method in methods:
        if method in ["hipporag"]:
            continue
        print(f"\n=== Starting with method: {method} ===")
        mydata = MyDataMain(
            mode=args.mode,
            method=method,
            device=args.device,
            use_multi_gpu=args.use_multi_gpu,
            chunk_mode=args.chunk_mode,
            output_dir=args.output_dir,
            persona_task_file=args.persona_task_file,
            emb_model_name=args.emb_model_name
        )
        
        # 각 persona에 대해 실행
        if method in ["standard", "random", "hipporag"] and args.mode == "indexing":
            mydata.run(-1)
        elif method in ["standard", "random", "hipporag"] and args.mode == "all":
            mydata.run(-1)
            for persona_index in indices:
                mydata.run(persona_index)
        else:
            for persona_index in indices:
                mydata.run(persona_index)
        
        print(f"\n=== Completed method: {method} ===")

if __name__ == "__main__":
    main() 