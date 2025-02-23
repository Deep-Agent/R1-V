# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

# fix transformers version(fix data type inconsistency)
pip install git+https://github.com/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519

# fix deepspeed version
pip install deepspeed==0.15.4