"""Finetuning hyperparameters: SFT vs RFT/GRPO."""

# SFT Config
SFT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
}

# GRPO Config
GRPO_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "sft_checkpoint": "",  # Path to SFT checkpoint (fill after SFT)
    "group_size": 4,  # G rollouts per prompt
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
    "kl_coeff": 0.1,  # KL penalty coefficient
    "kl_target": 0.01,  # Target KL divergence
    "clip_range": 0.2,
    "reward_weights": {
        "task_completion": 0.6,
        "step_efficiency": 0.2,
        "action_correctness": 0.2,
    },
    "fp16": True,
    "logging_steps": 5,
    "save_steps": 50,
}

# Data Config
DATA_CONFIG = {
    "train_file": "data/processed/mind2web_sft.jsonl",
    "eval_split": 0.1,
    "max_train_samples": 5000,
    "max_eval_samples": 500,
}
