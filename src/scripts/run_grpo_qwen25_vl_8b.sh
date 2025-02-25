#!/bin/bash

export WANDB_PROJECT="ocr_r1"
export DEBUG_MODE="true"

MODEL_PATH=checkpoints/Qwen/VL/Qwen2.5-VL-3B-Instruct
DATA_PATH=playground/DocVQA-R1

OUTPUT_DIR=outputs
export LOG_PATH="./logs/vllm_run_test_7.txt"
RUN_NAME=ocr1_test_8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="15001" \
    src/ocr_r1/train_grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --num_generations 6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels 200000 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true \
    --deepspeed src/scripts/zero3.json \
    --use_vllm True \
    # --num_processes 8 \