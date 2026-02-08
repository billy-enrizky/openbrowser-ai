"""Flow matching training hyperparameters: Discrete Flow Matching + GRPO."""

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

DATA_CONFIG = {
    "train_file": "data/processed/mind2web_flow.jsonl",
    "eval_split": 0.1,
    "max_train_samples": 5000,
}
