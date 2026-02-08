"""Flow matching training hyperparameters: Discrete Flow Matching + GRPO.

Two model backends:
    1. Small custom model (~30M params, byte-level) -- FlowVectorFieldEstimator
    2. LLM backbone (LLaDA-8B, ~8B params, QLoRA) -- FlowLLM

The LLM backend uses LLaDA (Large Language Diffusion with mAsking), a pre-trained
masked diffusion model. Unlike autoregressive LLMs, LLaDA uses bidirectional
attention and generates via iterative unmasking: tokens start fully masked and
are progressively revealed over multiple denoising steps.

This is architecturally distinct from the STAD68 AR approach (Qwen3-8B left-to-right
token generation), providing genuine model diversity between the two projects.
"""

import os

# --- Small custom flow model (legacy, ~30M params) ---
FLOW_MODEL_CONFIG = {
    "vocab_size": 256,  # Byte-level: matches tokenize_for_flow encoding (0-255)
    "hidden_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "max_seq_length": 512,
    "dropout": 0.1,
}

# --- LLM-backed flow model (LLaDA-8B with QLoRA) ---
# LLaDA: Large Language Diffusion with mAsking (GSAI-ML/LLaDA-8B-Base)
# - Bidirectional transformer encoder (no causal mask)
# - Native masked diffusion: mask token ID 126336, vocab size 126464
# - Must use AutoModel (not AutoModelForCausalLM) + trust_remote_code=True
# - No pre-quantized variant -- quantize on-the-fly with BitsAndBytesConfig
FLOW_LLM_CONFIG = {
    "model_name": "GSAI-ML/LLaDA-8B-Base",
    "base_model_name": "GSAI-ML/LLaDA-8B-Base",
    "trust_remote_code": True,
    "mask_token_id": 126336,
    "vocab_size": 126464,
    # QLoRA quantization (applied on-the-fly, no pre-quantized variant)
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    # LoRA -- proven targets from community LLaDA LoRA adapters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj"],
    # Generation
    "max_seq_length": 512,
    "max_new_tokens": 512,
    "num_denoising_steps": 64,  # LLaDA: best quality at steps == gen_length, 64 is a balance
    "generation_temperature": 0.7,
}

FLOW_SFT_CONFIG = {
    "learning_rate": 1e-4,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "10")),
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "num_ode_steps": 20,
    "sigma_min": 0.001,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 200,
}

# SFT config for LLM-backed flow model
FLOW_LLM_SFT_CONFIG = {
    "learning_rate": 2e-4,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "3")),
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 100,
}

FLOW_GRPO_CONFIG = {
    "group_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "batch_size": 8,
    "kl_coeff": 0.05,
    "clip_range": 0.2,
    "num_ode_steps": 10,
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
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "3")),
    "kl_coeff": 0.05,
    "clip_range": 0.2,
    "num_ode_steps": 10,
    "num_denoising_steps": 20,
    "fp16": True,
    "bf16": True,
    "logging_steps": 5,
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "max_new_tokens": 512,
    "reward_weights": {
        "task_completion": 0.6,
        "field_accuracy": 0.3,
        "execution_completeness": 0.1,
    },
}

DATA_CONFIG = {
    "train_file": os.environ.get("FLOW_TRAIN_FILE", "data/processed/formfactory_sft.jsonl"),
    "eval_split": 0.1,
    "max_train_samples": int(os.environ.get("MAX_TRAIN_SAMPLES", "5000")),
}
