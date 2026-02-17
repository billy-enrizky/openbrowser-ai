"""Flow matching training hyperparameters: Masked Diffusion + GRPO.

Two model backends:
    1. Small custom model (~30M params, byte-level) -- FlowVectorFieldEstimator
    2. LLM backbone (ReFusion 8B, ~8B params, QLoRA) -- Masked Diffusion LLM

The LLM backend uses ReFusion (GSAI-ML/ReFusion), a masked diffusion model
built on Qwen3. Unlike autoregressive LLMs, ReFusion uses slot-based parallel
decoding with iterative unmasking: response tokens start fully masked and
are progressively revealed via confidence-based scheduling.

ReFusion's forward() handles the masked diffusion loss internally:
    - Slot-based masking (slots of 4/8/16/32 tokens)
    - Hybrid loss = AR_loss (unmasked slots) + MDM_loss (masked slots / p_mask)
    - Requires prompt_lengths to separate prompt from response

This is architecturally distinct from the STAD68 AR approach (Qwen3-8B left-to-right
token generation), providing genuine model diversity between the two projects.
"""

import os

# --- Small custom flow model (~30M params, experimental) ---
FLOW_MODEL_CONFIG = {
    "vocab_size": 256,  # Byte-level: matches tokenize_for_flow encoding (0-255)
    "hidden_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "max_seq_length": 512,
    "dropout": 0.1,
}

# --- LLM-backed flow model (ReFusion 8B with QLoRA) ---
# ReFusion: Masked Diffusion LLM with parallel AR decoding (GSAI-ML/ReFusion)
# - Built on Qwen3 architecture (Qwen3ForCausalLM)
# - Uses AutoModelForCausalLM + trust_remote_code=True
# - Mask token ID: 151670, vocab size: 151671
# - Hybrid training: AR loss on unmasked slots + MDM loss on masked slots
# - Slot-based generation with KV cache reuse
FLOW_LLM_CONFIG = {
    "model_name": "GSAI-ML/ReFusion",
    "base_model_name": "GSAI-ML/ReFusion",
    "trust_remote_code": True,
    "mask_token_id": 151670,
    "vocab_size": 151671,
    # QLoRA quantization
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    # LoRA -- Qwen3 attention + MLP projections
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    # Generation
    "max_seq_length": 512,
    "max_new_tokens": 512,
    "num_denoising_steps": 64,
    "generation_temperature": 0.7,
    # ReFusion slot-based generation params
    "slot_size": 8,
    "slot_threshold": 0.9,
    "token_threshold": 0.9,
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
    "num_ode_steps": 20,
    "bf16": True,
    "logging_steps": 5,
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "max_new_tokens": 512,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}

# --- FS-DFM 1.3B (Apple, true discrete flow matching) ---
# Pre-trained on FineWeb-Edu, GPT-2 tokenizer (vocab=50257), DiT architecture
# Checkpoint: aminr8/FS-DFM -> DFM_checkpoint.pth (base model, FP32)
# Architecture: DDiTBlock with adaLN modulation, rotary embeddings, Poisson jump sampling
FSDFM_MODEL_CONFIG = {
    "hf_repo": "aminr8/FS-DFM",
    "checkpoint_filename": "DFM_checkpoint.pth",
    "hidden_size": 2048,
    "n_blocks": 21,
    "n_heads": 32,
    "cond_dim": 256,
    "mlp_ratio": 4,
    "vocab_size": 50257,  # GPT-2 tokenizer vocab (mask token not in embedding)
    "max_seq_length": 1024,
    "dropout": 0.1,
    # LoRA for fine-tuning (~5.5M trainable, 0.42% of 1.3B)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_target_layers": ["qw", "kw", "vw", "attn_out"],
    # Flow matching scheduler
    "scheduler_type": "polynomial",
    "scheduler_exponent": 2.0,
    "source_distribution": "uniform",
    # Generation
    "num_sampling_steps": 64,
    "generation_temperature": 1.0,
}

FSDFM_SFT_CONFIG = {
    "learning_rate": 2e-4,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "5")),
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 200,
    "grad_clip": 1.0,
}

ONLINE_FSDFM_GRPO_CONFIG = {
    "group_size": 2,
    "learning_rate": 5e-5,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "kl_coeff": 0.05,
    "clip_range": 0.2,
    "bf16": True,
    "logging_steps": 5,
    "grad_clip": 1.0,
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "num_sampling_steps": 64,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}

# --- Flow-GRPO for FS-DFM (discrete policy gradients, Liu et al. 2025) ---
# Adapts continuous Flow-GRPO (ODE-to-SDE + Gaussian log-probs) to the
# discrete Poisson jump process used by FS-DFM.  Per-step categorical
# log-probabilities are computed from the jump process, enabling PPO-style
# clipped policy gradients aligned with the actual generation trajectory.
# Reference: github.com/yifan123/flow_grpo
FLOW_GRPO_FSDFM_CONFIG = {
    "group_size": 4,                   # Increased from 2: G=2 gave zero advantages (both rollouts same reward)
    "learning_rate": 1e-5,            # Lower than SFT (2e-4) per PPO convention
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "kl_coeff": 0.04,                 # Matches reference geneval config
    "clip_range": 0.2,                # Standard PPO clip
    "adv_clip_max": 5.0,              # Advantage clipping (from reference)
    "bf16": True,
    "logging_steps": 5,
    "grad_clip": 1.0,
    "num_generation_steps": 64,        # Denoising steps (T=64, must match eval; T=32 still produces noise)
    "generation_temperature": 1.0,     # Increased from 0.7: old GRPO at 1.0 achieved 74% nonzero, 0.7 got 18%, 0.3 got 7%
    "num_sampled_timesteps": 8,        # Denoising reduction: sample K random steps from T-step trajectory (Flow-GRPO paper)
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}

# --- Flow-GRPO for ReFusion 8B (masked diffusion policy gradients) ---
# Adapts Flow-GRPO to ReFusion's iterative unmasking process.  Per-step
# log-probabilities are computed at newly-unmasked positions, enabling
# PPO-style clipped policy gradients aligned with the generation trajectory.
# Fixes the generation/optimization mismatch in the existing GRPO trainer
# which used autoregressive log-probs despite masked diffusion generation.
FLOW_GRPO_REFUSION_CONFIG = {
    "group_size": 4,                   # Increased from 2: G=2 gave zero advantages (both rollouts same reward)
    "learning_rate": 1e-5,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "kl_coeff": 0.04,
    "clip_range": 0.2,
    "adv_clip_max": 5.0,
    "bf16": True,
    "logging_steps": 5,
    "grad_clip": 1.0,
    "num_generation_steps": 64,        # Denoising steps (T=64, must match eval; T=20 produces garbled text)
    "generation_temperature": 1.0,     # Increased from 0.7: masked diffusion too deterministic at 0.7, near-identical rollouts
    "confidence_noise_std": 0.1,       # Noise on confidence scores for diverse position unmasking order
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}

DATA_CONFIG = {
    "train_file": os.environ.get("FLOW_TRAIN_FILE", "data/processed/formfactory_sft.jsonl"),
    "eval_split": 0.1,
    "max_train_samples": int(os.environ.get("MAX_TRAIN_SAMPLES", "5000")),
}
