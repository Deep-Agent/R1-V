r1_v_path=/workspace/xxx/github/R1-V
cd ${r1_v_path}

model_path=${r1_v_path}/output/train@geo170k/checkpoint-30
batch_size=4
output_path=${r1_v_path}/output/train@geo170k/eval/res@checkpoint-30.json
prompt_path=${r1_v_path}/src/eval/prompts/geoqa_test_prompts.jsonl
gpu_ids=0,1,2,3,4,5,6,7

python src/eval/test_qwen2vl_geoqa_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}
