import os
import re
import csv
import time
import json
import torch
import warnings
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue, cpu_count, set_start_method
from tqdm.auto import tqdm  # tqdm.auto를 사용하여 더 나은 진행 상황 표시
import threading
import requests
import random
import subprocess
import atexit
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.Custom import CustomSummarizationModel, CustomQAModel, CustomEmbeddingModel
# 토크나이저 병렬화 설정
os.environ["emb_TOKENIZERS_PARALLELISM"] = "false"

# 멀티프로세싱 시작 방식 설정
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # 이미 설정되어 있는 경우 무시

# 공통 설정
ROOT_DIR = "data"
TOP_K = 5

# Prompt 템플릿 파일 경로
PROMPT_DIR = os.path.join(ROOT_DIR, "prompt")
PROMPT_LLM_FILTERING = os.path.join(PROMPT_DIR, "mydata_llm_filtering.txt")
PROMPT_LLM_FILTERING_NP = os.path.join(PROMPT_DIR, "mydata_llm_filtering_np.txt")
PROMPT_LLM_FILTERING_WOKEEP = os.path.join(PROMPT_DIR, "mydata_llm_filtering_wokeep.txt")
PROMPT_LLM_FILTERING_WOSUM = os.path.join(PROMPT_DIR, "mydata_llm_filtering_wosum.txt")
PROMPT_LLM_FILTERING_P_NP = os.path.join(PROMPT_DIR, "mydata_llm_filtering_p_np.txt")
PROMPT_LLM_FILTERING_SYSTEM = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt.txt")
PROMPT_LLM_FILTERING_SYSTEM_EX = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_ex.txt")
PROMPT_LLM_FILTERING_USER = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt.txt")
PROMPT_LLM_SUMMARIZING = os.path.join(PROMPT_DIR, "mydata_llm_summarizing.txt")
PROMPT_LLM_SUMMARIZING_ONLY = os.path.join(PROMPT_DIR, "mydata_llm_summarizing_only_improved.txt")  # 개선된 프롬프트 사용
PROMPT_GENERATION = os.path.join(PROMPT_DIR, "mydata_generation.txt")
PROMPT_LLM_SUMMARIZING_ONLY_WOPREF = os.path.join(PROMPT_DIR, "mydata_llm_summarizing_only_wopref.txt")
PROMPT_LLM_SUMMARIZING_ENHANCED = os.path.join(PROMPT_DIR, "summarizing_cm_improved.txt")
PROMPT_LLM_SUMMARIZING_SYSTEM = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_systemprompt.txt")
PROMPT_LLM_SUMMARIZING_SYSTEM_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_systemprompt_ld.txt")
PROMPT_LLM_SUMMARIZING_USER = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_userprompt.txt")
PROMPT_LLM_SUMMARIZING_USER_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_userprompt_ld.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOSUM = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wosum.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOSUM_EX = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wosum_ex.txt")
PROMPT_LLM_FILTERING_SYSTEM_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_ld.txt")
PROMPT_LLM_FILTERING_USER_WOSUM = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_wosum.txt")
PROMPT_LLM_FILTERING_USER_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_ld.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOKEEP = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wokeep.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EX = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wokeep_ex.txt")
PROMPT_LLM_FILTERING_USER_WOKEEP = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_wokeep.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOFILTER = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wofilter.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOFILTER_EX = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wofilter_ex.txt")
PROMPT_LLM_FILTERING_USER_WOFILTER = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_wofilter.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wokeep_ld.txt")
PROMPT_LLM_FILTERING_USER_WOKEEP_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_wokeep_ld.txt")
PROMPT_LLM_FILTERING_SYSTEM_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_explicit_ld.txt")
PROMPT_LLM_FILTERING_USER_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_explicit_ld.txt")
PROMPT_LLM_FILTERING_SYSTEM_WOKEEP_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_systemprompt_wokeep_explicit_ld.txt")
PROMPT_LLM_FILTERING_USER_WOKEEP_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_userprompt_wokeep_explicit_ld.txt")
PROMPT_LLM_SUMMARIZING_SYSTEM_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_systemprompt_explicit_ld.txt")
PROMPT_LLM_SUMMARIZING_USER_EXPLICIT_LD = os.path.join(PROMPT_DIR, "enhanced/enhanced_sum_userprompt_explicit_ld.txt")

PROMPT_LLM_WQ_SYSTEM_PROMPT = os.path.join(PROMPT_DIR, "enhanced/wq_select_systemprompt.txt")
PROMPT_LLM_WQ_USER_PROMPT = os.path.join(PROMPT_DIR, "enhanced/wq_select_userprompt.txt")

PROMPT_LLM_WQ_CHECK_SYSTEM_PROMPT = os.path.join(PROMPT_DIR, "enhanced/wq_check_systemprompt.txt")
PROMPT_LLM_WQ_CHECK_USER_PROMPT = os.path.join(PROMPT_DIR, "enhanced/wq_check_userprompt.txt")


# Indexing 관련 설정
INDEXING_REPORT_FILE = "indexing_report.csv"
THRESHOLD = 0.45

# Generation 관련 설정
GENERATION_REPORT_FILE = "generation_report.csv"

# Evaluation 관련 설정
EVALUATION_REPORT_FILE = "evaluation_report.csv"
ERROR_TYPE_DIR = os.path.join(ROOT_DIR, "error_type")

def set_global_seed(seed):
    """모든 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

class MyDataUtils:
    # 클래스 변수로 시드 설정
    _seed = 42
    set_global_seed(_seed)
    
    def __init__(self, mode, method, device, use_multi_gpu, chunk_mode, output_dir, persona_task_file=None, emb_model_name="facebook/contriever", doc_mode="sample", vllm_server_url="http://localhost:8008/v1", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", index_type="hnsw", ssh_config=None, use_trust_align=False):
        """
        MyData 유틸리티 클래스 초기화
        
        Args:
            mode (str): 실행 모드 ('standard' 또는 'persona')
            method (str): 사용할 방법 ('standard', 'naive_p', 'cosine_only', 'random', 'hipporag')
            device (str): 사용할 디바이스 ('cuda' 또는 'cpu')
            use_multi_gpu (bool): 멀티 GPU 사용 여부
            chunk_mode (str): 청크 모드 ('wdoc' 또는 'wodoc')
            output_dir (str): 출력 디렉토리 경로
            persona_task_file (str, optional): persona task 파일 경로
            emb_model_name (str): 임베딩 모델 이름
            doc_mode (str): 문서 모드 ('total' 또는 'sample')
            vllm_server_url (str): vLLM 서버 URL
            llm_model_name (str): LLM 모델 이름
            index_type (str): FAISS 인덱스 타입 ('flat' 또는 'hnsw')
            ssh_config (dict, optional): SSH 설정 {'host': 'hostname', 'user': 'username', 'port': 22, 'key_file': '/path/to/key'}
        """
        self.mode = mode
        self.method = method
        if self.method == "raptor":
            custom_summarizer = CustomSummarizationModel()
            custom_qa = CustomQAModel()
            custom_embedding = CustomEmbeddingModel()
            custom_config = RetrievalAugmentationConfig(
                summarization_model=custom_summarizer,
                qa_model=custom_qa,
                embedding_model=custom_embedding,
            )
            self.raptor = RetrievalAugmentation(config=custom_config)
            self.raptor_save = os.path.join(ROOT_DIR, "corpus/raptor_tree")
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.chunk_mode = chunk_mode
        self.emb_model_name = emb_model_name
        self.doc_mode = doc_mode
        self.vllm_server_url = vllm_server_url
        self.llm_model_name = llm_model_name
        self.index_type = index_type
        self.ssh_config = ssh_config
        self.ssh_tunnel_process = None
        self.use_trust_align = use_trust_align
        # SSH 설정을 이용한 모델별 서버 URL 설정
        if ssh_config:
            model_config = None
            
            # 모델별 설정이 있는 경우
            if "models" in ssh_config and llm_model_name in ssh_config["models"]:
                model_config = ssh_config["models"][llm_model_name]
                print(f"🔗 Found SSH config for model: {llm_model_name}")
            # 기본 설정 사용
            elif "default" in ssh_config:
                model_config = ssh_config["default"]
                print(f"🔗 Using default SSH config for model: {llm_model_name}")
            # 단일 설정 (이전 버전 호환성)
            elif "host" in ssh_config:
                model_config = ssh_config
                print(f"🔗 Using single SSH config for model: {llm_model_name}")
            
            if model_config:
                local_port = model_config.get("local_port", 8009)
                self.vllm_server_url = f"http://localhost:{local_port}/v1"
                print(f"🔗 Using SSH-forwarded server: {self.vllm_server_url}")
                
                # SSH 터널링 자동 설정
                self.setup_ssh_tunnel(model_config)
            else:
                self.vllm_server_url = vllm_server_url
                print(f"🌐 No SSH config found, using standard server: {self.vllm_server_url}")
        else:
            self.vllm_server_url = vllm_server_url
            print(f"🌐 Using standard server for {llm_model_name}: {self.vllm_server_url}")
        
        # doc_mode와 chunk_mode에 따른 파일 경로 설정
        model_name_clean = emb_model_name.replace("/", "_")
        
        if doc_mode == "total":
            self.chunk_file = "data/corpus/full_chunks_with_doc.jsonl"
            self.embedding_file = f"data/corpus/full_embeddings_with_doc_{model_name_clean}.npy"
        elif doc_mode == "total_sw":
            self.chunk_file = "data/corpus/full_chunks_with_doc_sw.jsonl"
            self.embedding_file = f"data/corpus/full_sw_embeddings_with_doc_{model_name_clean}.npy"
        elif doc_mode == "sample_sw":
            self.chunk_file = "data/corpus/sampled_chunks_with_doc_sw.jsonl"
            self.embedding_file = f"data/corpus/sampled_sw_embeddings_with_doc_{model_name_clean}.npy"
        else:  # sample
            if chunk_mode == "wodoc":
                self.chunk_file = "data/corpus/sampled_chunks.jsonl"
                self.embedding_file = f"data/corpus/sampled_embeddings_{model_name_clean}.npy"
            elif chunk_mode == "wdoc":
                self.chunk_file = "data/corpus/sampled_chunks_with_doc.jsonl"
                self.embedding_file = f"data/corpus/sampled_embeddings_with_doc_{model_name_clean}.npy"
            else:
                raise ValueError(f"Invalid chunk_mode: {chunk_mode}. Must be either 'wodoc' or 'wdoc'")
        
        self.persona_task_file = f"data/{persona_task_file}"
        print(f"Persona task file: {self.persona_task_file}")
        
        # doc_mode에 따라 출력 디렉토리 설정
        if doc_mode == "total":
            self.output_dir = f"{output_dir}/total"
        elif doc_mode == "total_sw":
            self.output_dir = f"{output_dir}/total_sw"
        elif doc_mode == "sample_sw":
            self.output_dir = f"{output_dir}/sample_sw"
        else:  # sample
            self.output_dir = f"{output_dir}/sample"
        self.emb_tokenizer = None
        self.emb_model = None
        self.hf_emb_tokenizer = None
        self.hf_model = None
        
        # GPU 메모리 최적화를 위한 배치 사이즈 설정
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            self.num_gpus = torch.cuda.device_count()
            # 각 GPU의 메모리 크기에 따라 배치 사이즈 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                self.base_batch_size = 64
            elif gpu_memory >= 48e9:  # 48GB
                self.base_batch_size = 128
            elif gpu_memory >= 24e9:  # 24GB
                self.base_batch_size = 64
            elif gpu_memory >= 16e9:  # 16GB
                self.base_batch_size = 32
            else:
                self.base_batch_size = 16
            self.batch_size = self.base_batch_size * self.num_gpus
        else:
            self.num_gpus = 1
            # 단일 GPU의 경우에도 메모리 크기에 따라 배치 사이즈 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 80e9:  # 80GB
                self.batch_size = 256
            elif gpu_memory >= 48e9:  # 48GB
                self.batch_size = 128
            elif gpu_memory >= 24e9:  # 24GB
                self.batch_size = 64
            elif gpu_memory >= 16e9:  # 16GB
                self.batch_size = 32
            else:
                self.batch_size = 16
    
    def setup_ssh_tunnel(self, ssh_config):
        """SSH 터널링 설정"""
        try:
            host = ssh_config.get('host', 'localhost')
            user = ssh_config.get('user', 'root')
            port = ssh_config.get('port', 22)
            key_file = ssh_config.get('key_file', None)
            remote_port = ssh_config.get('remote_port', 8009)
            local_port = ssh_config.get('local_port', 8009)
            
            # SSH 설정 정보 출력
            print(f"🔗 SSH 설정 정보:")
            print(f"   Host: {host}")
            print(f"   User: {user}")
            print(f"   Port: {port}")
            print(f"   Key file: {key_file}")
            print(f"   Remote port: {remote_port}")
            print(f"   Local port: {local_port}")
            
            # 틸드 확장 처리
            if key_file:
                if key_file.startswith('~'):
                    key_file = os.path.expanduser(key_file)
                elif key_file.startswith('./'):
                    key_file = os.path.abspath(key_file)
                
                print(f"   Expanded key file path: {key_file}")
            
            # SSH 키 파일 존재 및 권한 확인
            if key_file:
                if not os.path.exists(key_file):
                    raise FileNotFoundError(f"SSH key file not found: {key_file}")
                
                # 키 파일 권한 확인
                file_stat = os.stat(key_file)
                file_mode = oct(file_stat.st_mode)[-3:]
                if file_mode != '600':
                    print(f"⚠️ SSH key file permissions: {file_mode} (recommended: 600)")
                    print(f"   Run: chmod 600 {key_file}")
                    # 권한 자동 수정 시도
                    try:
                        os.chmod(key_file, 0o600)
                        print(f"✅ SSH key file permissions fixed to 600")
                    except Exception as e:
                        print(f"❌ Failed to fix permissions: {e}")
                        raise
            
            print(f"🔗 Setting up SSH tunnel...")
            print(f"   Local port: {local_port}")
            print(f"   Remote: {user}@{host}:{port}")
            print(f"   Remote port: {remote_port}")
            
            # 기존 터널 종료 (포트 충돌 방지)
            self.cleanup_existing_tunnel(local_port)
            
            # SSH 연결 테스트
            print(f"🔗 Testing SSH connection...")
            test_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no', '-p', str(port)]
            if key_file:
                test_cmd.extend(['-i', key_file])
            test_cmd.extend([f'{user}@{host}', 'echo "SSH connection test successful"'])
            
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    print(f"✅ SSH connection test passed")
                else:
                    print(f"❌ SSH connection test failed:")
                    print(f"   stdout: {result.stdout}")
                    print(f"   stderr: {result.stderr}")
                    raise RuntimeError(f"SSH connection test failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("SSH connection test timed out")
            except Exception as e:
                raise RuntimeError(f"SSH connection test error: {e}")
            
            # SSH 명령 구성 (evaluation_with_different_llm.py 방식)
            ssh_cmd = [
                "ssh",
                "-i", key_file,
                "-L", f"{local_port}:localhost:{remote_port}",
                "-N",  # 명령어 실행하지 않음
                "-f",  # 백그라운드 실행
                f"{user}@{host}",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null"
            ]
            
            if port != 22:
                ssh_cmd.extend(["-p", str(port)])
            
            print(f"🔗 Starting SSH tunnel...")
            print(f"   Local port: {local_port} -> Remote port: {remote_port}")
            print(f"   Command: {' '.join(ssh_cmd)}")
            
            # SSH 터널링 실행
            self.ssh_tunnel_process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 터널 설정 대기 (evaluation_with_different_llm.py와 동일)
            time.sleep(3)
            
            # SSH 프로세스 상태 확인 (evaluation_with_different_llm.py 방식)
            if self.ssh_tunnel_process.poll() is None:
                print("✅ SSH tunnel process started successfully")
                
                # 포트 연결 테스트
                print(f"🔗 Testing local port connection on {local_port}...")
                port_test_attempts = 0
                max_port_attempts = 5
                
                while port_test_attempts < max_port_attempts:
                    if self.test_local_port(local_port):
                        print(f"✅ SSH tunnel established successfully on port {local_port}")
                        break
                    else:
                        port_test_attempts += 1
                        if port_test_attempts < max_port_attempts:
                            print(f"⏳ Port test attempt {port_test_attempts}/{max_port_attempts}, waiting...")
                            time.sleep(2)
                        else:
                            print(f"⚠️ Port {local_port} not responding after {max_port_attempts} attempts")
                            print("   SSH tunnel may need more time or there might be an issue")
                            # 계속 진행 (터널이 늦게 활성화될 수 있음)
                
            else:
                # 프로세스가 종료된 경우 (evaluation_with_different_llm.py에서는 이 경우를 에러로 처리하지 않음)
                stdout, stderr = self.ssh_tunnel_process.communicate()
                error_msg = stderr.decode('utf-8') if stderr else ""
                stdout_msg = stdout.decode('utf-8') if stdout else ""
                
                print(f"⚠️ SSH tunnel process terminated immediately:")
                print(f"   Exit code: {self.ssh_tunnel_process.returncode}")
                print(f"   stdout: {stdout_msg}")
                print(f"   stderr: {error_msg}")
                
                # Warning 메시지만 있는 경우는 정상으로 처리 (known_hosts 추가 메시지)
                if "Warning" in error_msg and "known hosts" in error_msg and self.ssh_tunnel_process.returncode == 0:
                    print("   This appears to be just a known hosts warning, checking if tunnel is actually working...")
                    
                    # 실제로 포트가 활성화되었는지 확인
                    time.sleep(2)
                    if self.test_local_port(local_port):
                        print(f"✅ SSH tunnel is actually working on port {local_port}")
                        self.ssh_tunnel_process = None  # 프로세스 추적 비활성화
                    else:
                        print("⚠️ SSH tunnel not responding on expected port")
                        print("   This might be normal - some tunnels take time to establish")
                        print("   Continuing with the assumption that tunnel will work...")
                        self.ssh_tunnel_process = None  # 프로세스 추적 비활성화
                else:
                    raise RuntimeError(f"SSH tunnel failed to establish: {error_msg}")
            
            # 프로세스 종료 시 터널 정리
            atexit.register(self.cleanup_ssh_tunnel)
            
        except Exception as e:
            print(f"❌ Failed to setup SSH tunnel: {str(e)}")
            print(f"   Please check:")
            print(f"   1. SSH key file exists and has correct permissions (600): {key_file}")
            print(f"   2. Server is accessible: {host}:{port}")
            print(f"   3. Remote port is open: {remote_port}")
            print(f"   4. SSH connection works: ssh -i {key_file} {user}@{host}")
            raise
    
    def cleanup_existing_tunnel(self, local_port):
        """기존 터널 정리 (포트 충돌 방지)"""
        try:
            # lsof 명령으로 포트 사용 중인 프로세스 찾기
            result = subprocess.run(
                ['lsof', '-ti', f':{local_port}'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], check=True)
                        print(f"🔗 Killed existing process on port {local_port}: PID {pid}")
                    except subprocess.CalledProcessError:
                        pass
                        
        except FileNotFoundError:
            # lsof 명령이 없는 경우 무시
            pass
        except Exception as e:
            print(f"⚠️ Error cleaning up existing tunnel: {e}")
    
    def test_local_port(self, port):
        """로컬 포트 연결 테스트"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def cleanup_ssh_tunnel(self):
        """SSH 터널 정리"""
        if self.ssh_tunnel_process:
            try:
                self.ssh_tunnel_process.terminate()
                self.ssh_tunnel_process.wait(timeout=5)
                print("🔗 SSH tunnel closed")
            except subprocess.TimeoutExpired:
                self.ssh_tunnel_process.kill()
                print("🔗 SSH tunnel forcibly closed")
            except Exception as e:
                print(f"⚠️ Error closing SSH tunnel: {str(e)}")
    
    def __del__(self):
        """소멸자에서 SSH 터널 정리"""
        self.cleanup_ssh_tunnel()
    
    def load_models(self):
        """모델과 토크나이저 로드"""
        # 임베딩 모델 로드
        print(f"Loading {self.emb_model_name} model...")
        self.emb_tokenizer = AutoTokenizer.from_pretrained(self.emb_model_name)
        self.emb_model = AutoModel.from_pretrained(self.emb_model_name).eval()
        
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {self.num_gpus} GPUs with batch size {self.batch_size}!")
            # 먼저 모델을 device로 이동
            self.emb_model = self.emb_model.to(self.device)
            # 그 다음 DataParallel 적용
            self.emb_model = DataParallel(
                self.emb_model,
                device_ids=list(range(self.num_gpus)),
                output_device=0,  # 메인 GPU
                dim=0  # 배치 차원
            )
            print(f"Embedding model distributed across {self.num_gpus} GPUs")
        else:
            self.emb_model = self.emb_model.to(self.device)
            print(f"Embedding model loaded on {self.device}")
        
        # LLM 모델 로드 (필요한 경우)
        if self.method in ["naive_p"] and self.mode in [""]:
            print("Loading LLM model...")
            set_seed(42)
            self.hf_emb_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.hf_emb_tokenizer.padding_side = 'left'  # 왼쪽 패딩으로 설정
            self.hf_emb_tokenizer.pad_token = self.hf_emb_tokenizer.eos_token
            
            # GPU 메모리 관리를 위한 설정
            torch.cuda.empty_cache()
            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                # 각 GPU에 모델을 분산하여 로드
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",  # 자동으로 GPU에 분산
                    use_cache=True,     # 캐시 사용
                    low_cpu_mem_usage=True  # CPU 메모리 사용 최소화
                )
                print(f"LLM model automatically distributed across {torch.cuda.device_count()} GPUs")
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16,
                    use_cache=True,
                    low_cpu_mem_usage=True
                ).to(self.device)
                print(f"LLM model loaded on {self.device}")
            self.hf_model.eval()
    
    def embed_texts(self, texts):
        all_embs = []
        # 배치 사이즈 최적화
        batch_size = self.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.emb_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.emb_model(**inputs)
                if isinstance(outputs.last_hidden_state, tuple):
                    embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(embeddings)
        return np.vstack(all_embs)
    
    def embed_query(self, query):
        inputs = self.emb_tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
            if isinstance(outputs.last_hidden_state, tuple):
                query_emb = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
            else:
                query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        return query_emb
    
    def call_llm(self, prompts, max_new_tokens=512):
        """LLM 호출 (배치 처리)"""
        try:
            # 배치 단위로 처리
            results = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Starting from v4.46, the `logits` model output will have the same type as the model",
                        category=UserWarning,
                    )
                    # 배치 전체를 한 번에 처리
                    formatted_prompts = [
                        self.hf_emb_tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True
                        ) for prompt in batch_prompts
                    ]
                    
                    # 배치 입력 준비
                    inputs = self.hf_emb_tokenizer(
                        formatted_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    ).to(self.device)
                    
                    # 생성 설정
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": 0.1,
                        "pad_token_id": self.hf_emb_tokenizer.eos_token_id,
                        "eos_token_id": self.hf_emb_tokenizer.eos_token_id
                    }
                    
                    # 배치 생성 (경고 메시지 제거)
                    with torch.no_grad():
                        outputs = self.hf_model.generate(
                            **inputs,
                            **generation_config
                        )
                    
                    # 배치 결과 디코딩
                    batch_results = [
                        self.hf_emb_tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    results.extend(batch_results)
            
            return results
            
        except Exception as e:
            print(f"Error in call_llm: {str(e)}")
            return None

    def generate_message_vllm(self, messages, system_prompt, max_tokens=512):
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.vllm_server_url}/chat/completions"
        
        # vLLM 서버 요구사항에 맞게 메시지 형식 수정
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        payload = {
            "model": self.llm_model_name,
            "messages": formatted_messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,  # top_k 비활성화
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "dtype": "float32"
        }
        
        for attempt in range(5):
            try:
                # timeout을 60초로 늘림
                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}/5] Request failed: {e}")
                if attempt < 4:  # 마지막 시도가 아니면 잠시 대기
                    # 재시도 간격을 지수적으로 증가 (1초, 2초, 4초, 8초)
                    wait_time = min(2 ** attempt, 10)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        raise RuntimeError("Failed to get response from vLLM server after 5 attempts")

    def parse_explanation_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        explanation_tag = soup.find("explanation")
        explanation = explanation_tag.text.strip() if explanation_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return explanation, answer

    def parse_preference_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        preference_tag = soup.find("preference")
        preference = preference_tag.text.strip() if preference_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return preference, answer
    
    def load_persona_data(self, persona_index):
        with open(self.persona_task_file, "r", encoding="utf-8") as f:
            personas = json.load(f)
        return next(p for p in personas if p["persona_index"] == persona_index)

    def load_persona_questions(self, file_path, persona_index):
        with open(file_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        for p in personas:
            if p["persona_index"] == persona_index:
                all_qs = []
                for block in p["preference_blocks"]:
                    pref = block["preference"]
                    for q in block["queries"]:
                        all_qs.append((pref, q["question"]))
                return all_qs
        raise ValueError(f"Persona index {persona_index} not found.")

    def retrieve_top_k(self, query, index, chunks, top_k=TOP_K):
        query_emb = self.embed_query(query)
        start_retrieval = time.time()
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time

    def retrieve_top_k_wq(self, query, preference, index, chunks, top_k=TOP_K):
        query_emb = self.embed_query(query)
        preference_emb = self.embed_query(preference)
        query_emb = query_emb + preference_emb
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)    # !
        start_retrieval = time.time()
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time
    
    def extract_preferences_from_response(self, response):
        soup = BeautifulSoup(response, "html.parser")
        preferences = [tag.text.strip() for tag in soup.find_all("preference")]
        return preferences

    def retrieve_top_k_wq_llm(self, query, preferences, index, chunks, top_k=TOP_K):
        system_prompt = self.load_prompt_template(PROMPT_LLM_WQ_SYSTEM_PROMPT)
        preferences_text = "\n".join(f" - {p}" for p in preferences)
        # preferences_text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(preferences))
        filled_user_prompt = self.load_prompt_template(PROMPT_LLM_WQ_USER_PROMPT).format(preferences=preferences_text, question=query)
        response = self.generate_message_vllm(
            messages=[{"role": "user", "content": filled_user_prompt}],
            system_prompt=system_prompt,
            max_tokens=1024
        )
        
        query_emb = self.embed_query(query)
        extracted_preferences = self.extract_preferences_from_response(response)
        for preference in extracted_preferences:
            if preference in preferences:
                query_emb += self.embed_query(preference)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)    # !
        start_retrieval = time.time()
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time

    def retrieve_top_k_wq_llm_2(self, query, preferences, index, chunks, top_k=TOP_K):
        """First-Level"""
        system_prompt = self.load_prompt_template(PROMPT_LLM_WQ_CHECK_SYSTEM_PROMPT)
        preferences_text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(preferences))
        filled_user_prompt = self.load_prompt_template(PROMPT_LLM_WQ_CHECK_USER_PROMPT).format(preferences=preferences_text, question=query)
        response = self.generate_message_vllm(
            messages=[{"role": "user", "content": filled_user_prompt}],
            system_prompt=system_prompt,
            max_tokens=1024
        )

        """Second-Level"""
        if "true" in response.lower():
            system_prompt = self.load_prompt_template(PROMPT_LLM_WQ_SYSTEM_PROMPT)
            filled_user_prompt = self.load_prompt_template(PROMPT_LLM_WQ_USER_PROMPT).format(preferences=preferences_text, question=query)
            response = self.generate_message_vllm(
                messages=[{"role": "user", "content": filled_user_prompt}],
                system_prompt=system_prompt,
                max_tokens=1024
            )
        
        query_emb = self.embed_query(query)
        extracted_preferences = self.extract_preferences_from_response(response)
        for preference in extracted_preferences:
            if preference in preferences:
                query_emb += self.embed_query(preference)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)    # !
        start_retrieval = time.time()
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time

    def load_prompt_template(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    def format_prompt(self, template, preference_text, chunk_text):
        return template.replace("{preference}", preference_text).replace("{chunk}", chunk_text)

    def save_jsonl(self, file_path, items):
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def save_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_csv(self, file_path, fieldnames, row, write_header=False):
        write_header = write_header or not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def parse_decision_and_reason(self, input_string):
        """LLM 응답에서 decision과 reason 파싱"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        return decision, reason

    def parse_decision_and_reason_preference(self, input_string):
        """LLM 응답에서 decision과 reason, preference 파싱"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        preference_tag = soup.find("relevant_preference")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        preference = preference_tag.text.strip() if preference_tag else ""
        
        # preference 텍스트 정리
        preference = self.clean_preference_text(preference)
        
        return decision, reason, preference

    def clean_preference_text(self, preference_text):
        """다양한 형태의 preference 텍스트에서 실제 선호 내용만 추출"""
        if not preference_text:
            return ""
        
        # 여러 줄로 되어 있는 경우 한 줄로 만들기
        preference_text = preference_text.strip()
        
        # 숫자만 있는 경우 (예: "1, 2" 또는 "5")
        if preference_text.replace(",", "").replace(" ", "").isdigit():
            return preference_text
        
        # "Preference X:" 형태 제거
        import re
        preference_text = re.sub(r'^Preference\s+\d+:\s*', '', preference_text, flags=re.IGNORECASE)
        
        # 앞에 숫자와 점이 있는 경우 제거 (예: "1. I prefer...")
        preference_text = re.sub(r'^\d+\.\s*', '', preference_text)
        
        # 따옴표 제거
        preference_text = preference_text.strip('"\'')
        
        return preference_text.strip()

    def map_preference_numbers_to_text(self, preference_text, preference_list):
        """숫자로 된 preference를 실제 텍스트로 매핑"""
        if not preference_text or not preference_list:
            return preference_text
        
        import re
        
        # 숫자들을 찾아서 실제 preference 텍스트로 매핑
        # "1, 2" -> [1, 2] 또는 "5" -> [5]
        numbers = re.findall(r'\d+', preference_text)
        
        if not numbers:
            return preference_text
        
        # 숫자를 실제 preference 텍스트로 변환
        mapped_preferences = []
        for num in numbers:
            try:
                index = int(num) - 1  # 1-based index를 0-based로 변환
                if 0 <= index < len(preference_list):
                    mapped_preferences.append(preference_list[index])
                else:
                    # 인덱스가 범위를 벗어나면 원래 숫자 그대로 사용
                    mapped_preferences.append(num)
            except ValueError:
                # 숫자가 아닌 경우 그대로 사용
                mapped_preferences.append(num)
        
        if mapped_preferences:
            return "; ".join(mapped_preferences)
        else:
            return preference_text

    def parse_summary(self, input_string):
        """LLM 응답에서 summary 파싱"""
        soup = BeautifulSoup(input_string, "html.parser")
        summary_tag = soup.find("summary")
        if summary_tag:
            return summary_tag.text.strip()
        else:
            # summary 태그가 없는 경우 fallback 처리
            text = input_string.strip()
            
            # 패턴 1: "Based on the provided user preferences..." 로 시작하는 경우
            if text.startswith("Based on the provided user preferences"):
                # "Document Chunk:" 또는 "Chunk:" 이후의 실제 내용 찾기
                chunk_match = re.search(r"(?:Document )?Chunk:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
                if chunk_match:
                    # 찾은 청크 내용을 반환 (\n\n에서 끊어짐)
                    chunk_content = chunk_match.group(1).strip()
                    return chunk_content
                else:
                    # "Document Chunk:" 또는 "Chunk:"가 없는 경우 원본 문장으로 반환    
                    return text
            
            # 그 외의 경우 전체 응답 반환 (짧은 경우)
            return None

    def process_chunk(self, chunk_text, preference_text, prompt_template, prompt_template_system=None, preference_list=None):
        filled_prompt = self.format_prompt(prompt_template, preference_text, chunk_text)
        try:
            if prompt_template_system is None:
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt="You are a helpful assistant for indexing document chunks."
                )
                decision, reason = self.parse_decision_and_reason(llm_response)
            else:
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt=prompt_template_system
                )
                decision, reason, preference = self.parse_decision_and_reason_preference(llm_response)
                
                # 숫자로 된 preference를 실제 텍스트로 매핑
                if preference_list and preference:
                    preference = self.map_preference_numbers_to_text(preference, preference_list)
            
            if decision == "":
                return None
            if prompt_template_system is None:
                return {
                    "chunk": chunk_text,
                    "decision": decision,
                    "reason": reason,
                    "status": "success"
                }
            else:
                return {
                    "chunk": chunk_text,
                    "decision": decision,
                    "reason": reason,
                    "relevant_preference": preference,
                    "status": "success"
                }
        except Exception as e:
            print(f"Failed to process chunk: {e}")
            return {
                "chunk": chunk_text,
                "decision": "Filter",  # 실패한 경우 기본적으로 필터
                "reason": f"LLM processing failed: {str(e)}",
                "status": "failed"
            }

    def process_chunk_per_preference(self, chunk_text, preference_list, prompt_template):
        """각 선호도별로 개별적으로 청크를 처리하고 결과를 종합"""
        per_preference_results = []
        
        # 각 선호도별로 개별 처리
        for i, preference in enumerate(preference_list):
            preference_text = f"- {preference}"
            filled_prompt = self.format_prompt(prompt_template, preference_text, chunk_text)
            
            try:
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt="You are a helpful assistant for indexing document chunks."
                )
                decision, reason = self.parse_decision_and_reason(llm_response)
                
                per_preference_results.append({
                    "preference_index": i,
                    "preference": preference,
                    "decision": decision if decision else "Filter",  # 빈 응답은 Filter로 처리
                    "reason": reason
                })
            except Exception as e:
                print(f"Failed to process chunk for preference {i}: {e}")
                per_preference_results.append({
                    "preference_index": i, 
                    "preference": preference,
                    "decision": "Filter",
                    "reason": f"LLM processing failed: {str(e)}"
                })
        
        # 결과 종합 및 최종 결정
        summarize_preferences = []
        keep_preferences = []
        filter_count = 0
        
        for result in per_preference_results:
            if result["decision"] == "Summarize":
                summarize_preferences.append(result["preference"])
            elif result["decision"] == "Keep As-Is":
                keep_preferences.append(result["preference"])
            elif result["decision"] == "Filter":
                filter_count += 1
        
        # 최종 결정 로직
        if summarize_preferences:
            # Summarize가 하나라도 있으면 Summarize (해당 선호도들을 모아서 사용)
            final_decision = "Summarize"
            relevant_preferences = summarize_preferences
            final_reason = f"Relevant to {len(summarize_preferences)} preference(s): {', '.join(summarize_preferences[:2])}{'...' if len(summarize_preferences) > 2 else ''}"
        elif keep_preferences:
            # Summarize가 없고 Keep이 하나라도 있으면 Keep
            final_decision = "Keep As-Is"
            relevant_preferences = keep_preferences
            final_reason = f"Directly relevant to {len(keep_preferences)} preference(s): {', '.join(keep_preferences[:2])}{'...' if len(keep_preferences) > 2 else ''}"
        else:
            # 모두 Filter이면 Filter
            final_decision = "Filter"
            relevant_preferences = []
            final_reason = "Not relevant to any user preferences"
        
        return {
            "chunk": chunk_text,
            "decision": final_decision,
            "reason": final_reason,
            "relevant_preferences": relevant_preferences,  # Summarize나 Keep에 관련된 선호도들
            "per_preference_results": per_preference_results,  # 각 선호도별 상세 결과
            "status": "success"
        }

    def process_chunk_with_resume(self, chunk_text, preference_text, prompt_template, prompt_template_system, preference_list, result_file, chunk_index):
        """
        실시간 저장 기능이 있는 청크 처리 함수 (JSONL 형식으로 append)
        """
        # 이미 처리된 청크인지 확인
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            if result.get("chunk_index") == chunk_index or result["chunk"] == chunk_text:
                                return result
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # 새로운 청크 처리
        filled_prompt = self.format_prompt(prompt_template, preference_text, chunk_text)
        try:
            llm_response = self.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt=prompt_template_system
            )
            decision, reason, preference = self.parse_decision_and_reason_preference(llm_response)
            
            # 숫자로 된 preference를 실제 텍스트로 매핑
            if preference_list and preference:
                preference = self.map_preference_numbers_to_text(preference, preference_list)
            
            if decision == "":
                return None
            
            result = {
                "chunk": chunk_text,
                "chunk_index": chunk_index,
                "decision": decision,
                "reason": reason,
                "relevant_preference": preference,
                "status": "success"
            }
            
            # JSONL 형식으로 파일 끝에 추가 (멀티스레딩 안전)
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            return result
            
        except Exception as e:
            print(f"Failed to process chunk {chunk_index}: {e}")
            result = {
                "chunk": chunk_text,
                "chunk_index": chunk_index,
                "decision": "Filter",  # 실패한 경우 기본적으로 필터
                "reason": f"LLM processing failed: {str(e)}",
                "status": "failed"
            }
            
            # 실패한 경우도 JSONL 형식으로 추가
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            return result

    def load_existing_results_with_resume(self, result_file):
        """
        기존 결과 파일에서 이미 처리된 청크들을 로드 (JSONL 형식)
        """
        if os.path.exists(result_file):
            try:
                existing_results = []
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_results.append(json.loads(line.strip()))
                return existing_results
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []

    def summarize_single(self, entry, summarizing_prompt, preference="N/A", summarizing_prompt_system=None):
        # entry가 딕셔너리인지 문자열인지 확인
        if isinstance(entry, dict):
            original_chunk = entry["chunk"]
            reason = entry.get("reason", "")
            if self.method == "per_pref":
                preference = entry.get("relevant_preferences", [])
                preference_text = "\n".join([f"- {p}" for p in preference])
            else:
                preference_text = preference
            
            if summarizing_prompt_system is None:
                filled_prompt = summarizing_prompt.replace("{preference}", preference_text).replace("{chunk}", original_chunk).replace("{reason}", reason)
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt="You are a helpful assistant tasked with summarizing document chunks."
                )
            else:
                preference_text = entry.get("relevant_preference", [])
                filled_prompt = summarizing_prompt.replace("{preference}", preference_text).replace("{chunk}", original_chunk).replace("{reason}", reason)
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt=summarizing_prompt_system
                )
            # <summary> 태그에서 내용 추출
            summarized_text = self.parse_summary(llm_response)
            if summarized_text is None:
                return {
                    "summarized": original_chunk,
                    "original": original_chunk,
                    "reason": reason
                }
            elif summarizing_prompt_system is not None:
                return {
                    "summarized": summarized_text,
                    "original": original_chunk,
                    "reason": reason,
                    "relevant_preference": preference_text
                }
            else:
                return {
                    "summarized": summarized_text,
                    "original": original_chunk,
                    "reason": reason
                }
        else:
            # entry가 문자열인 경우 (kept_chunks에서 온 경우)
            original_chunk = entry
            if preference == "N/A":
                filled_prompt = summarizing_prompt.replace("{chunk}", original_chunk)
            else:
                filled_prompt = summarizing_prompt.replace("{preference}", preference).replace("{chunk}", original_chunk)
            llm_response = self.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt="You are a helpful assistant tasked with summarizing document chunks."
            )
            # <summary> 태그에서 내용 추출
            summarized_text = self.parse_summary(llm_response)
            if summarized_text is None:
                return {
                    "summarized": original_chunk,
                    "original": original_chunk
                }
            else:
                return {
                    "summarized": summarized_text,
                    "original": original_chunk
                }


        
    def extract_likes_dislikes(self, preference):
        prompt = f"""
Given the following preference statement, extract the likes and dislikes in a structured format.

Preference: {preference}

Please respond in the following format:
<likes>
A clear, concise sentence describing what the person likes or prefers
</likes>
<dislikes>
A clear, concise sentence describing what the person dislikes or prefers less. If there are no dislikes mentioned, write "None"
</dislikes>

Note:
- If the preference uses "more interested in X than Y", X goes in likes and Y in dislikes
- If the preference uses "less interested in X than Y", X goes in dislikes and Y in likes
- Express likes and dislikes as complete, meaningful sentences
- Be specific and concise in your extraction
- If there are multiple likes/dislikes, combine them into coherent sentences
- If no dislikes are mentioned in the preference, explicitly write "None" in the dislikes section
"""
        # LLM 호출 및 결과 파싱
        response = self.generate_message_vllm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a helpful assistant tasked with extracting likes and dislikes from preference statements."
            )

        # 결과 파싱
        likes = re.search(r"<likes>(.*?)</likes>", response, re.DOTALL)
        dislikes = re.search(r"<dislikes>(.*?)</dislikes>", response, re.DOTALL)
        
        return {
            "likes": likes.group(1).strip() if likes else "",
            "dislikes": dislikes.group(1).strip() if dislikes else "None"
        }

    def parse_numbered_preferences(self, preference_text, preference_list):
        """
        번호가 포함된 선호 텍스트를 파싱하여 실제 선호 텍스트와 매칭
        
        Args:
            preference_text: 번호가 포함된 선호 텍스트 (예: "1. I am fascinated by Renaissance...\n5. I love visiting heritage sites...")
            preference_list: 원본 선호 텍스트 리스트
            
        Returns:
            matched_preferences: 매칭된 선호 텍스트들의 리스트
        """
        matched_preferences = []
        
        # 줄 단위로 분리
        lines = preference_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 번호가 포함된 형태인지 확인 (예: "1. ", "2. ", "5. ")
            if line[0].isdigit() and '. ' in line:
                # 번호와 텍스트 분리
                dot_index = line.find('. ')
                if dot_index != -1:
                    extracted_text = line[dot_index + 2:].strip()
                    
                    # 추출된 텍스트를 preference_list에서 찾기
                    for pref in preference_list:
                        if extracted_text == pref:
                            matched_preferences.append(pref)
                            break
                    else:
                        # 완전 일치하지 않으면 부분 일치 확인
                        for pref in preference_list:
                            if extracted_text in pref or pref in extracted_text:
                                matched_preferences.append(pref)
                                break
                        else:
                            print(f"Warning: Could not match preference text: '{extracted_text}'")
            else:
                # 번호가 없는 경우 직접 매칭
                for pref in preference_list:
                    if line == pref:
                        matched_preferences.append(pref)
                        break
                else:
                    # 부분 일치 확인
                    for pref in preference_list:
                        if line in pref or pref in line:
                            matched_preferences.append(pref)
                            break
                    else:
                        print(f"Warning: Could not match preference text: '{line}'")
        
        return matched_preferences

    def enhance_embeddings_with_preferences(self, embeddings, preference_embeddings, document_preference_mapping, preference_list):
        """
        문서 임베딩에 선호 임베딩을 더해서 방향을 조정
        
        Args:
            embeddings: 문서 임베딩 (numpy array)
            preference_embeddings: 선호 임베딩 (numpy array)
            document_preference_mapping: 문서별 관련 선호 매핑 정보 (list of dicts)
            preference_list: 선호 텍스트 리스트
        
        Returns:
            enhanced_embeddings: 향상된 임베딩 (numpy array)
        """
        print(f"Enhancing embeddings with preferences...")
        enhanced_embeddings = embeddings.copy()
        
        # 선호 텍스트를 인덱스로 매핑
        preference_to_idx = {pref: idx for idx, pref in enumerate(preference_list)}
        
        for i, doc_info in enumerate(document_preference_mapping):
            if 'relevant_preference' in doc_info and doc_info['relevant_preference']:
                relevant_prefs_raw = doc_info['relevant_preference']
                
                # 문자열인 경우 파싱
                if isinstance(relevant_prefs_raw, str):
                    # 번호가 포함된 형태인지 확인
                    if '\n' in relevant_prefs_raw and any(line.strip()[0].isdigit() for line in relevant_prefs_raw.split('\n') if line.strip()):
                        # 번호가 포함된 경우 파싱
                        relevant_prefs = self.parse_numbered_preferences(relevant_prefs_raw, preference_list)
                    else:
                        # 단일 선호인 경우
                        relevant_prefs = [relevant_prefs_raw]
                else:
                    # 이미 리스트인 경우
                    relevant_prefs = relevant_prefs_raw
                
                # 관련 선호 임베딩들 수집
                pref_embeddings = []
                for pref in relevant_prefs:
                    if pref in preference_to_idx:
                        pref_idx = preference_to_idx[pref]
                        pref_embeddings.append(preference_embeddings[pref_idx])
                    else:
                        print(f"Warning: Preference '{pref}' not found in preference_list")
                
                # 선호 임베딩이 있는 경우 평균을 구해서 더하기
                if pref_embeddings:
                    pref_embeddings = np.array(pref_embeddings)
                    # 여러 선호가 있는 경우 평균을 구함
                    avg_pref_embedding = np.mean(pref_embeddings, axis=0)
                    # 정규화
                    avg_pref_embedding = avg_pref_embedding / np.linalg.norm(avg_pref_embedding)
                    
                    # 문서 임베딩에 선호 임베딩을 더하기
                    enhanced_embeddings[i] = embeddings[i] + avg_pref_embedding
                    # 다시 정규화
                    enhanced_embeddings[i] = enhanced_embeddings[i] / np.linalg.norm(enhanced_embeddings[i])
        
        print(f"Enhanced {len([d for d in document_preference_mapping if 'relevant_preference' in d and d['relevant_preference']])} embeddings out of {len(document_preference_mapping)}")
        return enhanced_embeddings
