import json
import re

import torch
from math_verify import parse, verify
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

MODEL_PATH = "checkpoints/Qwen/VL/Qwen2-VL-2B-Instruct" # qwen2vl model or grpoed model on geoqa train
BSZ = 50 # reduce it if GPU OOM
OUTPUT_PATH = "./output/train@geo170k/eval/res@checkpoint-30.json"
PROMPT_PATH = "./src/eval/prompts/geoqa_test_prompts.jsonl"
IMAGE_ROOT = "./playground"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))


QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

messages = []

for i in data:
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{IMAGE_ROOT}/{i['image_path'][2:]}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=i['question'])
            }
        ]
    }]
    messages.append(message)




all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")





final_output = []
correct_number = 0

for input_example, model_output in zip(data,all_outputs):
    original_output = model_output
    ground_truth = input_example['ground_truth']
    model_answer = parse(original_output) 

    # Count correct answers
    if model_answer is not None and float(verify(model_answer,parse(ground_truth)))>0:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    
    try:
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer':str(model_answer[0]) if model_answer is not None else None,
            'is_correct':is_correct
        }

    except Exception as e:
        print("no answer parsed",e,model_answer)
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer':None,
            'is_correct':is_correct
        }



    final_output.append(result)


# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")





