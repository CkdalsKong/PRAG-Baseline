# MyData Pipeline

## 프로젝트 구조

### 데이터 및 실행 환경
```
/data/my_PRAG/
    └── baseline/
        ├── corpus/                    # 데이터 파일
        │   ├── sampled_chunks.jsonl
        │   ├── sampled_chunks_with_doc.jsonl
        │   ├── sampled_wiki_doc.jsonl
        │   └── sampled_embeddings.npy
        ├── error_type/                # 평가 기준
        │   ├── check_acknowledge.txt
        │   ├── check_hallucination.txt
        │   ├── check_helpful.txt
        │   └── check_violation.txt
        ├── output/                    # 결과 저장
        │   ├── standard/
        │   │   ├── kept.jsonl
        │   │   ├── faiss.index
        │   │   ├── embeddings.npy
        │   │   ├── gen_standard_{persona_index}.json
        │   │   └── eval_standard_{persona_index}.json
        │   ├── cosine_only_{persona_index}/
        │   │   ├── kept.jsonl
        │   │   ├── faiss.index
        │   │   ├── embeddings.npy
        │   │   ├── gen_cosine_only_{persona_index}.json
        │   │   └── eval_cosine_only_{persona_index}.json
        │   ├── naive_p_{persona_index}/
        │   │   ├── kept.jsonl
        │   │   ├── faiss.index
        │   │   ├── embeddings.npy
        │   │   ├── gen_naive_p_{persona_index}.json
        │   │   └── eval_naive_p_{persona_index}.json
        │   ├── indexing_report.csv
        │   ├── generation_report.csv
        │   └── evaluation_report.csv
        ├── prompt/                    # 프롬프트 템플릿
        │   ├── mydata_generation.txt
        │   ├── mydata_llm_filtering.txt
        │   └── mydata_llm_summarizing.txt
        └── final_persona_tasks.json   # Persona 태스크 정의

/home/ubuntu/changmin/Baseline/        # 소스 코드
    ├── run_vllm.sh                   # vLLM 서버 실행 스크립트
    ├── mydata_evaluation.py          # 평가 모듈
    ├── mydata_generation.py          # 생성 모듈
    ├── mydata_main.py               # 메인 실행 파일
    ├── mydata_indexing.py           # 인덱싱 모듈
    └── mydata_utils.py              # 유틸리티 모듈
```

## 환경 설정

### vLLM 서버 실행
```bash
# GPU 0,1을 사용하여 vLLM 서버 실행
./run_vllm.sh 0,1
```

## 실행 방법

### 기본 실행
```bash
python mydata_main.py --method [METHOD] --persona_index [INDEX] --mode [MODE] --chunk_mode [CHUNK_MODE] --output_dir [OUTPUT_DIR]
```

### 멀티 GPU 실행
```bash
CUDA_VISIBLE_DEVICES=0,1 python mydata_main.py --method [METHOD] --persona_index [INDEX] --mode [MODE] --chunk_mode [CHUNK_MODE] --output_dir [OUTPUT_DIR] --use_multi_gpu
```

## 파라미터 설명

### 필수 파라미터
- `--method`: 실행할 방법 선택
  - `naive_p`: Naive Persona 방식
  - `standard`: Standard 방식
  - `cosine_only`: Cosine Only 방식
  - `all`: 모든 방식 순차 실행
  - 예시: `--method naive_p` 또는 `--method all`

- `--persona_index`: Persona 인덱스 선택
  - `0-9`: 특정 Persona 인덱스
  - `all`: 모든 Persona 순차 실행
  - 예시: `--persona_index 0` 또는 `--persona_index all`

- `--mode`: 실행할 모드 선택
  - `indexing`: 인덱싱만 실행
  - `generation`: 생성만 실행
  - `evaluation`: 평가만 실행
  - `all`: 모든 모드 순차 실행
  - 예시: `--mode indexing` 또는 `--mode all`

- `--chunk_mode`: 청크 모드 선택
  - `wodoc`: 문서 정보 없는 청크 사용
  - `wdoc`: 문서 정보 포함된 청크 사용
  - 예시: `--chunk_mode wodoc`

- `--output_dir`: 출력 디렉토리 지정
  - 예시: `--output_dir output_1`

### 선택 파라미터
- `--device`: 사용할 GPU 디바이스 (기본값: "cuda:0")
  - 예시: `--device cuda:0`

- `--use_multi_gpu`: 멀티 GPU 사용 여부 (플래그)
  - 예시: `--use_multi_gpu`

## 실행 예시

### 단일 방법, 단일 Persona 실행
```bash
python mydata_main.py --method naive_p --persona_index 0 --mode all --chunk_mode wodoc --output_dir output_1
```

### 모든 방법, 모든 Persona 실행
```bash
python mydata_main.py --method all --persona_index all --mode all --chunk_mode wodoc --output_dir output_1
```

### 멀티 GPU로 실행
```bash
CUDA_VISIBLE_DEVICES=0,1 python mydata_main.py --method naive_p --persona_index all --mode all --chunk_mode wodoc --output_dir output_1 --use_multi_gpu
```

### 특정 모드만 실행
```bash
python mydata_main.py --method standard --persona_index all --mode indexing --chunk_mode wdoc --output_dir output_1
``` 