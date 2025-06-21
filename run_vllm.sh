#!/bin/bash

# 입력 인자로 GPU ID (예: 0,1)
GPUS=$1

if [ -z "$GPUS" ]; then
  echo "사용법: sh run_vllm.sh <GPU_IDs>"
  echo "예시: sh run_vllm.sh 0,1"
  exit 1
fi

# GPU 설정
export CUDA_VISIBLE_DEVICES=$GPUS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# GPU 개수 계산
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

source ~/miniforge3/etc/profile.d/conda.sh
conda activate hipporag

huggingface-cli login --token $HUGGINGFACE_TOKEN

# vLLM 서버 실행
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size $NUM_GPUS \
  --max_model_len 4096 \
  --gpu-memory-utilization 0.95 \
  --port 8006 \
  --seed 42
