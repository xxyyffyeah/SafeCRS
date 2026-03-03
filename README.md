# SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems

This repository contains the training and evaluation pipeline for **SafeCRS**, a safety-aware framework for LLM-based conversational recommender systems. SafeCRS combines **Safe-SFT** (Safe Supervised Fine-Tuning) with **Safe-GDPO** (Safe Group reward-Decoupled normalization Policy Optimization) to jointly optimize recommendation quality and personalized safety alignment.

> **SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems**
> Haochang Hao\*, Yifan Xu\*, Xinzhuo Li, Yingqiang Ge, Lu Cheng
> University of Illinois at Chicago, University of Illinois at Urbana-Champaign, Amazon

Building upon [Rank-GRPO](https://arxiv.org/abs/2506.05889) (Zhu et al., 2025), this work extends the framework with personalized safety constraints.

## Key Features

- **Personalized Safety Alignment**: Respects user-specific content sensitivities (trauma triggers, phobias, age-appropriateness) during recommendation
- **SafeRec Benchmark**: First user-centric safety benchmark for CRS with 20 sensitivity traits and oracle-grounded risk scoring
- **Two-Stage Training**: Safe-SFT for safety-aware data curation + Safe-GDPO for multi-reward optimization
- **96.5% Safety Improvement**: Reduces safety violations from 25% to <1% while maintaining recommendation quality

## Environment Setup

```bash
# Option 1: Create from environment file (recommended)
conda env create -f environment_safe.yml
conda activate safe

# Option 2: Manual installation
conda create -n safe python=3.13
conda activate safe
conda install -c nvidia cuda-toolkit=13.0.2
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install transformers==4.57.3 trl==0.27.2 peft==0.18.0 accelerate==1.12.0
pip install vllm==0.13.0 deepspeed==0.18.4 bitsandbytes==0.49.0
pip install wandb datasets

```

### Key Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| Python | 3.13.11 | Runtime |
| PyTorch | 2.9.0 | Deep learning framework |
| TRL | 0.27.2 | Transformer Reinforcement Learning |
| vLLM | 0.13.0 | Fast LLM inference |
| CUDA | 13.0 | GPU acceleration |

## Project Structure

```
SafeCRS/
├── train_sft_safe.py              # Safe-SFT training
├── train_grpo_safe.py             # Safe-GDPO training
├── libs/
│   ├── safe_reward_funcs.py       # Relevance, safety, count reward functions
│   ├── safety_oracle.py           # SafetyOracle for violation checking
│   ├── safety_filter.py           # Safety filtering utilities
│   ├── constraint_injector.py     # Constraint injection for prompts
│   ├── metrics.py                 # Evaluation metrics
│   ├── utils.py                   # Common utilities
│   └── trl/
│       └── rank_grpo_trainer.py   # Custom RankGRPOTrainer with GDPO support
└── evaluate/
    ├── eval_sft_val_safe.py       # Safe-SFT validation evaluation
    ├── eval_grpo_val.py           # GRPO validation evaluation
    └── eval_grpo_test.py          # Final test evaluation
```

## Datasets

### SafeRec Benchmark

SafeRec extends Reddit-V2 with personalized safety annotations:

| Component | Description |
|-----------|-------------|
| **Safety Oracle** | IMDb Parent Guide + DoesTheDogDie fusion for movies; ESRB for games |
| **20 Sensitivity Traits** | Anti-gore, Kid-safety, Self-harm sensitive, Horror avoider, etc. |
| **Latent Trait Injection** | Natural language constraints injected into prompts |

### Data Splits

| Split | Samples | Time Window |
|-------|---------|-------------|
| Train | ~19,086 | ≤ 2022-10 |
| Val   | ~1,127  | 2022-11 |
| Test  | ~1,212  | 2022-12 |

## Training Pipeline

### Stage 1: Safe-SFT

Safe-SFT fine-tunes the base model on SafeRec data where:
1. Ground-truth movies violating user constraints are filtered
2. Constraint descriptions are injected into prompts

```bash
torchrun --nproc_per_node=2 train_sft_safe.py \
    --dataset_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --bf16
```

### Stage 2: Safe-GDPO

Safe-GDPO refines the policy using three reward signals with decoupled normalization:

| Reward | Formula | Description |
|--------|---------|-------------|
| **Relevance** | `r_rel[k] = hit[k]` | Binary hit at rank k |
| **Safety** | `r_safe[k] = -λ·v[k]·d[k]` | Rank-discounted penalty for violations |
| **Count** | `r_cnt = ±λ_cnt` | Encourages target recommendation count |

#### TRL GDPOTrainer

```bash
python train_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_grpo_dataset \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --sft_model_path ./results/meta-llama/Llama-3.2-3B-Instruct/checkpoint-800 \
    --sft_is_lora \
    --catalog_path gt_catalog_complete.pkl \
    --advantage_mode gdpo \
    --lr 1e-6 \
    --kl_beta 1e-3 \
    --lambda_safe 1.0 \
    --penalty_safe 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --save_steps 200 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --bf16
```

### GRPO vs GDPO Mode

| Mode | `--advantage_mode` | Advantage Computation |
|------|-------------------|----------------------|
| **GRPO** | `grpo` | Sum rewards → Normalize: `A = norm(Σ wᵢrᵢ)` |
| **GDPO** | `gdpo` | Normalize → Sum: `A = Σ wᵢnorm(rᵢ)` |

GDPO prevents dense safety rewards from drowning sparse relevance signals.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of ground-truth items in top-K |
| **NDCG@K** | Normalized discounted cumulative gain |
| **SVR@K** | Safety Violation Ratio (↓ better) |
| **S-DCG@K** | Position-weighted safety violations (↓ better) |

### Commands

```bash
# Safe-SFT validation
python evaluate/eval_sft_val_safe.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --model_root ./results/meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --catalog_path gt_catalog_complete.pkl

# GRPO test evaluation
python evaluate/eval_grpo_test.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --model_root ./results/safe_grpo/meta-llama/Llama-3.2-3B-Instruct_gdpo_... \
    --checkpoint 1000 \
    --catalog_path gt_catalog_complete.pkl
```

## Results

### SafeMovie Benchmark

| Model | Recall@10 | NDCG@10 | SVR@10 (↓) |
|-------|-----------|---------|------------|
| GPT-4 (zero-shot) | 0.1128 | 0.0664 | 43.59% |
| GPT-5.2 (zero-shot) | 0.1379 | 0.0815 | 33.69% |
| **SafeCRS (Llama-3.1-8B)** | 0.1111 | 0.0737 | **0.87%** |
| **SafeCRS (Qwen2.5-0.5B)** | 0.0922 | 0.0597 | **0.06%** |

SafeCRS reduces safety violations by **96.5%** while maintaining competitive recommendation quality.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage_mode` | `grpo` | `grpo` or `gdpo` |
| `--lambda_safe` | 1.0 | Safety penalty weight |
| `--penalty_safe` | 1.0 | Penalty magnitude per violation |
| `--risk_threshold` | 0.66 | Safety Oracle threshold |
| `--lambda_count` | 0.0 | Count reward weight (0 = disabled) |
| `--target_count` | 10 | Target number of recommendations |
| `--reward_weights` | None | Weights for [relevance, safety, count] |
| `--use_lora` | False | Enable LoRA training |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA scaling factor |

## Citation

If you find this work helpful, please cite:

```bibtex
@article{hao2026safecrs,
  title={SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems},
  author={Hao, Haochang and Xu, Yifan and Li, Xinzhuo and Ge, Yingqiang and Cheng, Lu},
  journal={arXiv preprint},
  year={2026}
}

@article{zhu2025rankgrpo,
  title={Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning},
  author={Zhu, Yaochen and Steck, Harald and Liang, Dawen and He, Yinhan and Ostuni, Vito and Li, Jundong and Kallus, Nathan},
  journal={arXiv preprint arXiv:2510.20150},
  year={2025}
}
```

## License

This project is for research purposes. Please refer to the original datasets (Reddit-V2, IMDb, DoesTheDogDie) for their respective licenses.

## Acknowledgments

This work builds upon:
- [Rank-GRPO](https://github.com/yaochen-zhu/Rank-GRPO) by Netflix Research
- [GDPO](https://arxiv.org/abs/2601.05242) by NVlabs
- [TRL](https://github.com/huggingface/trl) by Hugging Face
