"""Flow matching training hyperparameters: Discrete Flow Matching + GRPO."""

import os

FLOW_MODEL_CONFIG = {
    "vocab_size": 32000,
    "hidden_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "max_seq_length": 256,
    "dropout": 0.1,
}

FLOW_SFT_CONFIG = {
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "num_ode_steps": 20,  # Euler steps for ODE solver
    "sigma_min": 0.001,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 200,
}

FLOW_GRPO_CONFIG = {
    "group_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "batch_size": 8,
    "kl_coeff": 0.05,
    "clip_range": 0.2,
    "num_ode_steps": 10,  # Fewer steps for faster rollouts
    "reward_weights": {
        "task_completion": 0.6,
        "step_efficiency": 0.2,
        "action_correctness": 0.2,
    },
    "fp16": True,
    "logging_steps": 5,
}

ONLINE_FLOW_GRPO_CONFIG = {
    "group_size": 2,  # Reduced from 4 -- browser execution is slower
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "kl_coeff": 0.05,
    "clip_range": 0.2,
    "num_ode_steps": 10,
    "fp16": True,
    "logging_steps": 5,
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "reward_weights": {
        "task_completion": 0.6,
        "field_accuracy": 0.3,
        "execution_completeness": 0.1,
    },
}

DATA_CONFIG = {
    "train_file": os.environ.get("FLOW_TRAIN_FILE", "data/processed/mind2web_flow.jsonl"),
    "eval_split": 0.1,
    "max_train_samples": int(os.environ.get("MAX_TRAIN_SAMPLES", "5000")),
}
