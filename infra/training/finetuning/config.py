"""Finetuning hyperparameters: SFT (QLoRA) and GRPO on Qwen3-8B."""

import os

# Base model for merging LoRA adapters and serving
BASE_MODEL_NAME = "Qwen/Qwen3-8B"

# SFT Config -- QLoRA on Qwen3-8B
SFT_CONFIG = {
    "model_name": "unsloth/Qwen3-8B-bnb-4bit",
    "base_model_name": BASE_MODEL_NAME,
    # QLoRA quantization
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",
    # Training
    "learning_rate": 2e-4,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "2")),
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
}

# GRPO Config
GRPO_CONFIG = {
    "model_name": "unsloth/Qwen3-8B-bnb-4bit",
    "base_model_name": BASE_MODEL_NAME,
    "sft_checkpoint": os.environ.get("SFT_CHECKPOINT_PATH", ""),
    # QLoRA quantization (same as SFT)
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    # GRPO-specific
    "group_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
    "max_new_tokens": 512,
    "kl_coeff": 0.1,
    "kl_target": 0.01,
    "clip_range": 0.2,
    "reward_weights": {
        "task_completion": 0.6,
        "step_efficiency": 0.2,
        "action_correctness": 0.2,
    },
    # LoRA (applied on top of SFT adapter)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",
    "bf16": True,
    "logging_steps": 5,
    "save_steps": 50,
}

# Online GRPO Config -- browser execution on FormFactory
ONLINE_GRPO_CONFIG = {
    **GRPO_CONFIG,
    "group_size": 2,  # Reduced from 4 -- browser execution is slower
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "reward_weights": {
        "task_completion": 0.6,
        "field_accuracy": 0.3,
        "execution_completeness": 0.1,
    },
}

# Data Config
DATA_CONFIG = {
    "train_file": os.environ.get("TRAIN_FILE", "data/processed/formfactory_sft.jsonl"),
    "eval_split": 0.1,
    "max_train_samples": int(os.environ.get("MAX_TRAIN_SAMPLES", "5000")),
    "max_eval_samples": 500,
}

# S3 Config for checkpoint persistence
S3_CONFIG = {
    "checkpoint_bucket": os.environ.get(
        "S3_CHECKPOINT_BUCKET", "openbrowser-eval-results-529206289231"
    ),
    "checkpoint_prefix": os.environ.get(
        "S3_CHECKPOINT_PREFIX", "training/checkpoints"
    ),
    "region": os.environ.get("AWS_REGION", "ca-central-1"),
}
