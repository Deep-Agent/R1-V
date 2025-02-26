#!/bin/bash

export WANDB_PROJECT="ocr_r1"
export DEBUG_MODE="true"

TIMESTAMP=$(date '+%y%m%d_%H%M%S')
PROJECT_NAME="Qwen25VL_7B@DocVQA@${TIMESTAMP}"

MODEL_PATH=checkpoints/Qwen/VL/Qwen2.5-VL-7B-Instruct
# MODEL_PATH=checkpoints/Qwen/VL/Qwen2-VL-2B-Instruct
DATA_PATH=playground/DocVQA-R1

# 256*28*28
MIN_PIXELS="100352"
# 576*28*28
MAX_PIXELS="451584"

RUN_NAME=$PROJECT_NAME
OUTPUT_DIR="outputs/${PROJECT_NAME}"
export LOG_PATH="logs/${PROJECT_NAME}.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="15001" \
    src/ocr_r1/train_grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels $MAX_PIXELS \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed src/scripts/zero3.json \
    # --use_vllm True \
    # --num_processes 8 \
