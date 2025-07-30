import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.nn.parallel import DataParallel
import argparse
from transformers import DPRQuestionEncoder, DPRContextEncoder
import gc
import time

def load_chunks(file_path):
    """JSONL 파일에서 청크 로드"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
    return chunks

def load_models(model_name):
    """임베딩 모델과 토크나이저 로드"""
    print(f"Loading {model_name} model...")
    is_nvidia_model = model_name.startswith("nvidia")

    if model_name.startswith("dpr"):
        # DPR 모델 로드
        if model_name == "dpr-question":
            model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        else:  # dpr-context
            model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif is_nvidia_model:
        max_mem = { 
            0: "45GiB", 
            1: "45GiB", 
            2: "45GiB", 
            3: "45GiB"
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_mem
        ).eval()
        device = None
        batch_size = 16
        return tokenizer, model, device, batch_size
    else:
        # 기존 임베딩 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
    
    # GPU 메모리 관리를 위한 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 배치 사이즈 설정
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs!")
        # 각 GPU의 메모리 크기에 따라 배치 사이즈 조정
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory >= 80e9:  # 80GB
            base_batch_size = 512
        elif gpu_memory >= 48e9:  # 48GB
            base_batch_size = 1024
        elif gpu_memory >= 24e9:  # 24GB
            base_batch_size = 32
        elif gpu_memory >= 16e9:  # 16GB
            base_batch_size = 16
        else:
            base_batch_size = 8
        batch_size = base_batch_size * num_gpus
        
        # 모델을 device로 이동 후 DataParallel 적용
        model = model.to(device)
        model = DataParallel(
            model,
            device_ids=list(range(num_gpus)),
            output_device=0,  # 메인 GPU
            dim=0  # 배치 차원
        )
        print(f"Model distributed across {num_gpus} GPUs with batch size {batch_size}")
    else:
        model = model.to(device)
        # 단일 GPU의 경우에도 메모리 크기에 따라 배치 사이즈 조정
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory >= 80e9:  # 80GB
            batch_size = 512
        elif gpu_memory >= 48e9:  # 48GB
            batch_size = 256
        elif gpu_memory >= 24e9:  # 24GB
            batch_size = 32
        elif gpu_memory >= 16e9:  # 16GB
            batch_size = 16
        else:
            batch_size = 16
        print(f"Model loaded on {device} with batch size {batch_size}")
    
    return tokenizer, model, device, batch_size

def generate_embeddings(chunks, tokenizer, model, device, batch_size=256, model_name="facebook/contriever"):
    """청크들의 임베딩 생성"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        
        # DPR 모델일 때만 max_length 제한 적용
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt"
        }
        if model_name.startswith("dpr"):
            tokenizer_kwargs["max_length"] = 512
        
        inputs = tokenizer(
            batch, 
            **tokenizer_kwargs
        )
        if device is not None:  # NV 모델일 경우 device=None
            inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 모델별 출력 처리
            if model_name.startswith("dpr"):  # DPR 모델
                embeddings = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'last_hidden_state'):  # facebook/contriever와 같은 모델
                if isinstance(outputs.last_hidden_state, tuple):
                    embeddings = outputs.last_hidden_state[0][:, 0, :].cpu().numpy()
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif isinstance(outputs, dict):  # sentence-transformers와 같은 모델
                if 'sentence_embedding' in outputs:
                    embeddings = outputs['sentence_embedding'].cpu().numpy()
                elif 'sentence_embeddings' in outputs:  # NVIDIA 모델용
                    embeddings = outputs['sentence_embeddings'].cpu().numpy()
                elif 'last_hidden_state' in outputs:
                    embeddings = outputs['last_hidden_state'][:, 0, :].cpu().numpy()
                else:
                    raise ValueError(f"Unexpected model output format: {outputs.keys()}")
            else:
                raise ValueError(f"Unsupported model output type: {type(outputs)}")
                
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/contriever", 
                      help="임베딩 모델 이름 (facebook/contriever, dpr-context, nvidia/NV-Embed-v2, princeton-nlp/sup-simcse-roberta-large)")
    parser.add_argument("--root_dir", type=str, default="../data/corpus", help="청크 파일 및 출력 디렉토리 경로")
    parser.add_argument("--chunk_mode", type=str, required=True, choices=["wodoc", "wdoc"], 
                      help="Chunk mode: 'wodoc' for chunks without document info, 'wdoc' for chunks with document info")
    parser.add_argument("--doc_mode", type=str, required=True, choices=["total", "sample", "related", "related2", "sample_sw", "total_sw"], 
                      help="Document mode: 'total' for all documents, 'sample' for sampled documents, 'related' for sampled related documents")
    args = parser.parse_args()

    # 모델 이름에서 파일명 생성
    model_name_clean = args.model_name.replace("/", "_")

    # 1. 청크 로드
    print("Loading chunks...")

    # doc_mode와 chunk_mode에 따른 파일 경로 설정
    if args.doc_mode == "total":
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "full_chunks_with_doc.jsonl")
        else:  # wodoc
            chunk_file = os.path.join(args.root_dir, "full_chunks_with_doc.jsonl")  # 문서 단위이므로 동일
    elif args.doc_mode == "total_sw":
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "full_chunks_with_doc_sw.jsonl")
        else:  # wodoc
            print("미구현")
            return
    elif args.doc_mode == "related":
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "sampled_related_chunks_with_doc.jsonl")
        else:  # wodoc
            print("미구현")
            return
    elif args.doc_mode == "related2":
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "sampled_related_chunks_with_doc2.jsonl")
        else:  # wodoc
            print("미구현")
            return
    elif args.doc_mode == "sample_sw":
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "sampled_chunks_with_doc_sw.jsonl")
        else:  # wodoc
            print("미구현")
            return
    else:  # sample
        if args.chunk_mode == "wdoc":
            chunk_file = os.path.join(args.root_dir, "sampled_chunks_with_doc.jsonl")
        else:  # wodoc
            chunk_file = os.path.join(args.root_dir, "sampled_chunks.jsonl")  

    # 파일 존재 확인
    if not os.path.exists(chunk_file):
        print(f"Error: File {chunk_file} not found!")
        print("Please run process_wiki.py first to generate the required files.")
        return

    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    
    # 2. 모델 로드
    tokenizer, model, device, batch_size = load_models(args.model_name)
    
    # 3. 임베딩 생성
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, tokenizer, model, device, batch_size, args.model_name)
    
    # 4. 임베딩 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 5. 결과 저장
    # chunk_mode에 따른 파일명 설정
    if args.chunk_mode == "wdoc":
        chunk_mode = "with_doc_"
    else:
        chunk_mode = ""
    
    # doc_mode에 따른 파일명 설정
    if args.doc_mode == "total":
        output_file = os.path.join(args.root_dir, f"full_embeddings_{chunk_mode}{model_name_clean}.npy")
    elif args.doc_mode == "related":
        output_file = os.path.join(args.root_dir, f"sampled_related_embeddings_{chunk_mode}{model_name_clean}.npy")
    elif args.doc_mode == "related2":
        output_file = os.path.join(args.root_dir, f"sampled_related2_embeddings_{chunk_mode}{model_name_clean}.npy")
    elif args.doc_mode == "sample_sw":
        output_file = os.path.join(args.root_dir, f"sampled_sw_embeddings_{chunk_mode}{model_name_clean}.npy")
    elif args.doc_mode == "total_sw":
        output_file = os.path.join(args.root_dir, f"full_sw_embeddings_{chunk_mode}{model_name_clean}.npy")
    else:  # sample
        output_file = os.path.join(args.root_dir, f"sampled_embeddings_{chunk_mode}{model_name_clean}.npy")
    
    print(f"Saving {len(embeddings)} embeddings to {output_file}")
    np.save(output_file, embeddings)
    print("Done!")

if __name__ == "__main__":
    main() 
