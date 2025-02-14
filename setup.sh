conda create -n r1-v python=3.11 
conda activate r1-v

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support
pip install vllm==0.7.2

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef