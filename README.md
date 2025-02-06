# R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3

![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/counting_star.png)

> ### Roadmap for R1-V
> We are building a general framework for RLVR in VLM. We believe in the power of **trenches** and **longtermism**.
>
> Our Interest: General Vision-Language Intelligence & Visual/GUI Agent
> 
> Our Goal: 🔄 Algorithm Enhancement ⚡ Efficiency Optimization 🎯 Task Diversity 🌲 Impactful Open Source Research. 
>
> Welcome Ideas and Contribution. Stay tuned!


1. We firstly reveal that **Reinforcement Learning with Verifiable Rewards (RLVR)** outperforms chain-of-thought supervised fine-tuning (CoT-SFT) in both **effectiveness and out-of-distribution (OOD) robustness** for vision language models.

2. In our experiment, we **incentivize** VLMs to learn **generalizable** visual counting abilities, rather than overfitting to the training set.

3. The 2B model outperforms the 72B model in OOD tests within just **100** training steps.

4. The training was conducted on 8 A100 GPUs for **30 minutes, costing $2.62**.


[🤗 Train Dataset: CLEVR-70k](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train)

[🤗 R1-Distilled Visual Reasoning Dataset](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1)

**Contributors:** [Liang Chen](https://github.com/chenllliang) · [Lei Li](https://lilei-nlp.github.io) · [Haozhe Zhao](https://haozhezhao.github.io/) · [Yifan Song](https://github.com/Yifan-Song793) · [Vinci](https://github.com/0xvincii)

---

### Updates

- 2025-02-06: We upload the evaluation script and polish the README. We are writing a blog post summarizing the statistics, findings and underexplored questions. 
- 2025-02-03: We upload the training codebase.
- 2025-02-03: We curate and upload some verified Deepseek-R1 visual reasoning traces with some special tricks (see `R1-V/src/distill_r1/`). Current training code does not rely on it, feel free to explore.
- 2025-02-03: We release the R1-V repo.

### For contributors
- Our top development priority is addressing the issues marked with `help wanted` labels, and we welcome ideas/PRs from the community to help solve them.

---


![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/ood.png)

![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/super_ood.png)

![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/training.png)

![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/curves.png)


## Setup

```bash
bash setup.sh
```


## Training

```bash
cd src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \
    --dataset_name <PATH-TO-DATASET-In-Repo> \  #https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true

```

> [!WARNING] 
> To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.

## Evaluation

![image](https://raw.githubusercontent.com/Deep-Agent/R1-V/main/images/train-test.png)

We provide the example script to evaluate OOD counting performance on a subset of SuperCLEVR within 1 minute. You can also modify the script and dataset to test on your own dataset.



```bash
cd ./src/eval
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip

# change the model path in the script
python test_qwen2vl_counting_superclevr.py 

# tested scores: 
# Qwen2VL-2B-Instruct: 48.0%
# Qwen2VL-2B-Instruct-GRPO-100step: 82.5%
```



## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) (our initial codebase), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR) for providing open source resources and help for us to build the project.

[![Star History Chart](https://api.star-history.com/svg?repos=Deep-Agent/R1-V&type=Timeline)](https://star-history.com/#Deep-Agent/R1-V&Timeline)

## Citation

```bib
@misc{chen2025r1v,
  author       = {Chen, Liang and Li, Lei and Zhao, Haozhe and Song, Yifan and Vinci},
  title        = {R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3},
  howpublished = {\url{https://github.com/Deep-Agent/R1-V}},
  note         = {Accessed: 2025-02-02},
  year         = {2025}
}
```




