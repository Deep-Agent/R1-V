


model_path=checkpoints/Qwen/VL/Qwen2-VL-2B-Instruct
batch_size=4
output_path=./output/train@geo170k/eval/res@checkpoint-30.json
prompt_path=./src/eval/prompts/geoqa_test_prompts.jsonl
image_root=./playground
gpu_ids=0,1,2,3

python src/eval/test_qwen2vl_geoqa_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids} \
    --image_root ${image_root}