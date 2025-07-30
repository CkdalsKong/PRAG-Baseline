import os
import json
import time
import argparse
import numpy as np
import faiss
from mydata_utils import MyDataUtils
from mydata_indexing import MyDataIndexing
from mydata_generation import MyDataGeneration
from mydata_evaluation import MyDataEvaluation

class MyDataMain:
    def __init__(self, mode="all", method="all", device="cuda:0", use_multi_gpu=False, chunk_mode="wodoc", output_dir=None, persona_task_file=None, emb_model_name="facebook/contriever", doc_mode="sample", vllm_server_url="http://localhost:8008/v1", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", index_type="hnsw", ssh_config=None, use_trust_align=False):
        self.mode = mode
        self.method = method
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.chunk_mode = chunk_mode
        self.output_dir = output_dir
        self.persona_task_file = persona_task_file
        self.emb_model_name = emb_model_name
        self.doc_mode = doc_mode
        self.vllm_server_url = vllm_server_url
        self.llm_model_name = llm_model_name
        self.index_type = index_type
        self.ssh_config = ssh_config
        self.use_trust_align = use_trust_align
        # 유틸리티 클래스 초기화 (한 번만)
        self.utils = MyDataUtils(
            mode=mode,
            method=method,
            device=device,
            use_multi_gpu=use_multi_gpu,
            chunk_mode=chunk_mode,
            output_dir=output_dir,
            persona_task_file=persona_task_file,
            emb_model_name=emb_model_name,
            doc_mode=doc_mode,
            vllm_server_url=vllm_server_url,
            llm_model_name=llm_model_name,
            index_type=index_type,
            ssh_config=ssh_config,
            use_trust_align=use_trust_align
        )
        
        # 각 단계별 처리 클래스 초기화 (한 번만)
        self.indexing = MyDataIndexing(self.utils)
        self.generation = MyDataGeneration(self.utils)
        self.evaluation = MyDataEvaluation(self.utils)
        
        # 🆕 공통 리소스 캐싱을 위한 속성들
        self._models_loaded = False
        self._chunks_cache = None
        self._embeddings_cache = None
        self._related_chunks_cache = None
        self._related_embeddings_cache = None
    
    def _load_common_resources(self):
        """공통 리소스를 한 번만 로딩하여 캐시"""
        if self.method == "raptor":
            return
        if self._models_loaded:
            print("✅ Common resources already loaded, using cache...")
            return
            
        print("🔄 Loading common resources...")
        start_time = time.time()
        
        # 1. 모델 로딩 (한 번만)
        print("📱 Loading models...")
        self.utils.load_models()
        
        # 2. 청크 데이터 로딩 (한 번만)
        print("📄 Loading chunks...")
        with open(self.utils.chunk_file, "r", encoding="utf-8") as f:
            self._chunks_cache = [json.loads(line)["text"] for line in f]
        
        # 3. 임베딩 로딩 (한 번만)
        print("🔢 Loading embeddings...")
        self._embeddings_cache = np.load(self.utils.embedding_file)
        
        # 4. 관련 청크 및 임베딩 로딩 (필요한 경우)
        model_name_clean = self.emb_model_name.replace("/", "_")
        if self.persona_task_file == "data/final_persona_tasks.json" and self.doc_mode == "sample":
            print("📄 Loading related chunks (final_persona_tasks)...")
            with open("data/corpus/sampled_related_chunks_with_doc.jsonl", "r", encoding="utf-8") as f:
                self._related_chunks_cache = [json.loads(line)["text"] for line in f]
            self._related_embeddings_cache = np.load(f"data/corpus/sampled_embeddings_with_doc_{model_name_clean}.npy")
            
            # 청크와 임베딩 결합
            self._chunks_cache.extend(self._related_chunks_cache)
            self._embeddings_cache = np.vstack((self._embeddings_cache, self._related_embeddings_cache))
            
        elif self.persona_task_file == "data/final_persona_tasks2.json" and self.doc_mode == "sample":
            print("📄 Loading related chunks (final_persona_tasks2)...")
            with open("data/corpus/sampled_related_chunks_with_doc2.jsonl", "r", encoding="utf-8") as f:
                self._related_chunks_cache = [json.loads(line)["text"] for line in f]
            self._related_embeddings_cache = np.load(f"data/corpus/sampled_related2_embeddings_with_doc_{model_name_clean}.npy")
            
            # 청크와 임베딩 결합
            self._chunks_cache.extend(self._related_chunks_cache)
            self._embeddings_cache = np.vstack((self._embeddings_cache, self._related_embeddings_cache))
        
        self._models_loaded = True
        load_time = time.time() - start_time
        print(f"✅ Common resources loaded in {load_time:.2f}s")
        print(f"   📊 Chunks: {len(self._chunks_cache)}")
        print(f"   📊 Embeddings: {self._embeddings_cache.shape}")
    
    def get_cached_resources(self):
        """캐시된 공통 리소스 반환"""
        if not self._models_loaded:
            self._load_common_resources()
        
        return {
            "chunks": self._chunks_cache,
            "embeddings": self._embeddings_cache,
            "models_loaded": True
        }
    
    def run_batch_processing(self, persona_indices):
        """여러 persona를 배치로 처리"""
        print(f"\n🚀 Starting batch processing for {len(persona_indices)} personas...")
        
        # 공통 리소스를 한 번만 로딩
        self._load_common_resources()
        
        # 각 persona 처리
        for persona_index in persona_indices:
            self.run_single_persona(persona_index)
    
    def run_single_persona(self, persona_index):
        """단일 persona 처리 (캐시된 리소스 사용)"""
        if persona_index == -1:
            print(f"\n=== Processing standard method {self.method} ===")
        else:
            print(f"\n=== Processing persona {persona_index} with method {self.method} ===")
        
        # 출력 디렉토리 설정
        if self.method in ["standard", "random", "random_1", "random_01", "hipporag"]:
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}")
        elif self.method[-3:] == "_wq":
            method_dir = os.path.join(self.utils.output_dir, f"{self.method[:-3]}/{persona_index}")
        elif self.method[-2:] == "wq":
            method_dir = os.path.join(self.utils.output_dir, f"{self.method[:-2]}/{persona_index}")
        elif self.method[-3:] == "wql":
            method_dir = os.path.join(self.utils.output_dir, f"{self.method[:-4]}/{persona_index}")
        else:
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}/{persona_index}")
        os.makedirs(method_dir, exist_ok=True)
        
        # 캐시된 공통 리소스 가져오기
        cached_resources = self.get_cached_resources()
        
        # 1. Indexing
        if self.mode in ["indexing", "all"] and self.method[-2:] != "wq" and self.method[-3:] != "wql":
            print("\n1. Starting indexing...")
            if self.index_type == "flat":
                faiss_index_path = os.path.join(method_dir, f"index_flat_{self.emb_model_name.replace('/', '_')}.faiss")
            else:
                faiss_index_path = os.path.join(method_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")
            if os.path.exists(faiss_index_path):
                print(f"✅ Indexing already completed. Skipping...")
            else:
                self.indexing.run_indexing_with_cache(persona_index, cached_resources)
                print(f"✅ Indexing completed. Results saved to {method_dir}")
        
        # 2. Generation
        if self.mode in ["generation", "all"] and persona_index != -1:
            print("\n2. Starting generation...")
            if self.index_type == "flat":
                if self.method[-2:] == "wq":
                    gen_file = os.path.join(method_dir, f"gen_wq_flat_{persona_index}.json")
                elif self.method[-3:] == "wql":
                    gen_file = os.path.join(method_dir, f"gen_wql_flat_{persona_index}.json")
                else:
                    gen_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")
            else:
                if self.method[-2:] == "wq":
                    gen_file = os.path.join(method_dir, f"gen_wq_{persona_index}.json")
                elif self.method[-3:] == "wql":
                    gen_file = os.path.join(method_dir, f"gen_wql_{persona_index}.json")
                elif self.use_trust_align:
                    gen_file = os.path.join(method_dir, f"gen_{self.method}_trustalign_{persona_index}.json")
                else:
                    gen_file = os.path.join(method_dir, f"gen_{self.method}_{persona_index}.json")
            if os.path.exists(gen_file):
                print(f"✅ Generation already completed. Skipping...")
            else:
                self.generation.run_generation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Generation completed. Results saved to {gen_file}")
        
        # 3. Evaluation
        if self.mode in ["evaluation", "all"] and persona_index != -1:
            print("\n3. Starting evaluation...")
            if self.index_type == "flat":
                if self.method[-2:] == "wq":
                    eval_file = os.path.join(method_dir, f"eval_wq_flat_{persona_index}.json")
                elif self.method[-3:] == "wql":
                    eval_file = os.path.join(method_dir, f"eval_wql_flat_{persona_index}.json")
                else:
                    eval_file = os.path.join(method_dir, f"eval_{self.method}_flat_{persona_index}.json")
            else:
                if self.method[-2:] == "wq":
                    eval_file = os.path.join(method_dir, f"eval_wq_{persona_index}.json")
                elif self.method[-3:] == "wql":
                    eval_file = os.path.join(method_dir, f"eval_wql_{persona_index}.json")
                elif self.use_trust_align:
                    eval_file = os.path.join(method_dir, f"eval_{self.method}_trustalign_{persona_index}.json")
                else:
                    eval_file = os.path.join(method_dir, f"eval_{self.method}_{persona_index}.json")
            if os.path.exists(eval_file):
                print(f"✅ Evaluation already completed. Skipping...")
            else:
                self.evaluation.run_evaluation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Evaluation completed. Results saved to {eval_file}")
        
        if persona_index == -1:
            print(f"\n=== Completed standard method ===")
        else:
            print(f"\n=== Completed persona {persona_index} ===")
    
    # 🔄 기존 run 메서드는 하위 호환성을 위해 유지
    def run(self, persona_index):
        """기존 방식 (하위 호환성용)"""
        self.run_single_persona(persona_index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["wokeep_like_emdir_wql", "wokeep_emdir_wql", "wokeep_ld_wq", "wokeep_explicit_ld_wq", "enhanced_prompt_wokeep_wq", "raptor", "wokeep_like_emdir_wq", "wokeep_like_emdir", "sum_embedding", "wokeep_ld_emdirwq", "wokeep_ld_emdir", "wokeep_explicitld_emdirwq", "wokeep_explicitld_emdir", "wokeep_explicit_ld", "explicit_ld", "wokeep_emdir_wq", "wokeep_ld", "wokeep_emdir_ex", "wokeep_emdir", "like_dislike", "em_dir", "em_dir_ex", "enhanced_prompt_wofilter_ex", "enhanced_prompt_wofilter", "enhanced_prompt_wokeep_ex", "enhanced_prompt_wokeep", "enhanced_prompt_wosum_ex", "enhanced_prompt_wosum", "enhanced_prompt_ex", "enhanced_prompt", "per_pref", "enhanced_sum", "sum_only_wopref", "llm_p_np", "naive_p", "naive_p_np", "wokeep_p", "wosum_p", "sum_only_p","standard", "cosine_only", "random", "random_1", "random_01", "hipporag", "pref_cluster_filter", "score_p", "naive_p_all", "all"], help="Method type: 'naive_p', 'standard', 'cosine_only', or 'all'")
    parser.add_argument("--persona_index", type=str, required=True, help="Persona index (0-10) or 'all'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--mode", type=str, required=True, choices=["indexing", "generation", "evaluation", "all"], help="Mode to run: 'indexing', 'generation', 'evaluation', or 'all'")
    parser.add_argument("--use_multi_gpu", action="store_true", help="멀티 GPU 사용 여부")
    parser.add_argument("--chunk_mode", type=str, required=True, choices=["wodoc", "wdoc"], help="Chunk mode: 'wodoc' for chunks without document info, 'wdoc' for chunks with document info")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--persona_task_file", type=str, required=True, help="Path to the persona task file")
    parser.add_argument("--emb_model_name", type=str, default="facebook/contriever", help="임베딩 모델 이름")
    parser.add_argument("--doc_mode", type=str, required=True, choices=["sample", "sample_sw", "total", "total_sw"], help="Document mode: 'sample' for sample documents, 'full' for full documents")
    parser.add_argument("--vllm_server_url", type=str, default="8008", help="vLLM server URL or port number (e.g., 8006 or http://localhost:8008/v1)")
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLM model name")
    parser.add_argument("--index_type", type=str, default="hnsw", help="FAISS index type: 'hnsw' for HNSW, 'flat' for Flat, 'ivf' for IVF")
    parser.add_argument("--use_trust_align", action="store_true", help="trust_align 모델 사용 여부")
    # SSH 설정 파일 경로
    parser.add_argument("--ssh_config_file", type=str, default=None, help="SSH 설정 파일 경로 (JSON 형식)")
    
    args = parser.parse_args()
    
    if args.use_trust_align:
        if args.mode in ["all", "indexing"]:
            print("❌ trust_align 모델은 generation, evaluation 모드에서만 사용할 수 있습니다.")
            exit(1)
        elif args.mode == "generation":
            args.llm_model_name = "declare-lab/trustalign_llama3_8b"

    # SSH 설정 로드
    ssh_config = None
    if args.ssh_config_file and os.path.exists(args.ssh_config_file):
        try:
            with open(args.ssh_config_file, 'r', encoding='utf-8') as f:
                ssh_config = json.load(f)
            print(f"🔗 SSH configuration loaded from: {args.ssh_config_file}")
            print(f"   Host: {ssh_config.get('host', 'N/A')}")
            print(f"   User: {ssh_config.get('user', 'N/A')}")
            print(f"   Port: {ssh_config.get('port', 22)}")
        except Exception as e:
            print(f"❌ Error loading SSH config file: {e}")
            ssh_config = None
    elif args.ssh_config_file:
        print(f"⚠️ SSH config file not found: {args.ssh_config_file}")
    
    # vLLM 서버 URL 처리: 포트 번호만 입력된 경우 전체 URL 구성
    vllm_server_url = args.vllm_server_url
    if vllm_server_url.isdigit():
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    elif not vllm_server_url.startswith("http"):
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    
    # Persona 인덱스 설정
    if args.persona_task_file == "final_persona_tasks.json":
        indices = list(range(10)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.persona_task_file == "prefeval_persona_tasks_final.json":
        indices = list(range(19)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.persona_task_file == "prefeval_persona_tasks_final2.json":
        indices = list(range(29)) if args.persona_index == "all" else [int(args.persona_index)]
    else:
        indices = list(range(20)) if args.persona_index == "all" else [int(args.persona_index)]
    
    # 실행할 방법들 설정
    methods = ["naive_p", "wosum_p", "standard", "cosine_only", "random"] if args.method == "all" else [args.method]
    
    # 각 방법에 대해 실행
    for method in methods:
        if method in ["hipporag"]:
            continue
        print(f"\n=== Starting with method: {method} ===")
        
        # MyDataMain 인스턴스 생성 (한 번만)
        mydata = MyDataMain(
            mode=args.mode,
            method=method,
            device=args.device,
            use_multi_gpu=args.use_multi_gpu,
            chunk_mode=args.chunk_mode,
            output_dir=args.output_dir,
            persona_task_file=args.persona_task_file,
            emb_model_name=args.emb_model_name,
            doc_mode=args.doc_mode,
            vllm_server_url=vllm_server_url,
            llm_model_name=args.llm_model_name,
            index_type=args.index_type,
            ssh_config=ssh_config,
            use_trust_align=args.use_trust_align
        )
        
        # 각 persona에 대해 실행
        if method in ["standard", "random", "random_1", "random_01", "hipporag", "raptor"]:
            if args.mode == "indexing":
                # standard/random 방법은 persona-independent하므로 -1로 한 번만 실행
                mydata.run_single_persona(-1)
            elif args.mode == "all":
                # indexing은 한 번만, generation/evaluation은 각 persona별로
                mydata.run_single_persona(-1)  # indexing
                if indices:
                    mydata.run_batch_processing(indices)  # generation/evaluation
            else:
                # generation/evaluation만 실행하는 경우
                if indices:
                    mydata.run_batch_processing(indices)
        else:
            # persona-dependent 방법들은 모든 persona에 대해 배치 처리
            if indices:
                mydata.run_batch_processing(indices)
        
        print(f"\n=== Completed method: {method} ===")

if __name__ == "__main__":
    main()
