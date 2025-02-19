export WANDB_PROJECT="r1-v"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

QWEN_PATH=checkpoints/Qwen/Qwen2.5-VL-3B-Instruct
HF_DATASET=playground/Clevr_CoGenT_TrainA_70K_Complex
OUTPUT_DIR="outputs_nonvllm"
RUN_NAME="r1v_wo_vllm"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node="6" \
    --nnodes="1" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --max_prompt_length 512 \
    --max_prompt_length 1024 \
    --temperature 1.0 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true