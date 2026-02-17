# ReFusion Flow-GRPO Design

## Problem

The existing ReFusion GRPO trainer (`online_flow_llm_grpo_trainer.py`) has a generation/optimization mismatch: it generates via iterative unmasking (masked diffusion) but optimizes using per-token autoregressive log-probabilities. This means the policy gradient signal is misaligned with the actual generation process.

Flow-GRPO (Liu et al., 2025) fixes this by computing per-step log-probabilities aligned with the denoising trajectory.

## Approach: Trajectory-Recording Unmasking with Per-Step Masked Log-Probs

Adapted from the FS-DFM Flow-GRPO implementation (`fsdfm_flow_grpo_trainer.py`), which itself adapts the continuous Flow-GRPO algorithm to discrete flow matching.

### Per-Step Log-Probability Definition

At each denoising step `t` of iterative unmasking:
1. ReFusion sees `[prompt | partially_masked_response]`
2. It predicts tokens for all masked positions
3. Top-k most confident masked positions are unmasked
4. **Log-prob for step t** = sum of `log softmax(logits)[pos, token]` over the newly-unmasked positions only

This focuses the policy gradient on the positions where the model actually made decisions.

## Architecture

### 1. Trajectory Data Structures (`flow_llm_model.py`)

```python
@dataclass
class UnmaskingTrajectoryStep:
    step_index: int                    # Which denoising step (0..T-1)
    masked_state: torch.Tensor         # [B, L_c + L_r] token IDs (with mask tokens)
    attention_mask: torch.Tensor       # [B, L_c + L_r] attention mask
    newly_unmasked_indices: list[list[int]]  # [B][k] indices into response portion
    unmasked_tokens: list[list[int]]   # [B][k] token IDs chosen

@dataclass
class UnmaskingTrajectory:
    steps: list[UnmaskingTrajectoryStep]
    final_tokens: torch.Tensor         # [B, L_r] fully unmasked response
    condition_length: int              # L_c (prompt length)
```

### 2. Trajectory-Recording Generation (`flow_llm_model.py`)

New method `FlowLLM.generate_with_trajectory()`:
- Same iterative unmasking logic as `FlowLLM.generate()`
- At each step, records the masked state and which positions were unmasked
- Returns `UnmaskingTrajectory` instead of just final tokens

### 3. Per-Step Log-Prob Computation (`flow_llm_model.py`)

New function `compute_unmasking_step_log_prob()`:
- Forward pass ReFusion with the recorded masked state (no labels, no prompt_lengths)
- Extract response logits: `logits[:, L_c:, :]`
- Compute `log_softmax` at newly-unmasked positions
- Sum log-probs over those positions -> scalar per-step log-prob
- Runs WITH gradients for current policy, WITHOUT for old/reference

### 4. Flow-GRPO Trainer (`refusion_flow_grpo_trainer.py`)

New file mirroring `fsdfm_flow_grpo_trainer.py` structure:

- **Phase 1 - Generation**: G rollouts via `generate_with_trajectory()` (T=10 steps)
- **Phase 2 - Execution**: Parse actions, execute in headless browser against FormFactory
- **Phase 3 - Advantages**: Group-relative advantages with outlier clipping
- **Phase 4 - Policy Gradient**: For each rollout, iterate over ALL T trajectory steps:
  - `policy_log_prob` = `compute_unmasking_step_log_prob(policy_model, step)` (with grads)
  - `old_log_prob` = `compute_unmasking_step_log_prob(policy_model, step)` (detached)
  - `ref_log_prob` = `compute_unmasking_step_log_prob(ref_model, step)` (detached)
  - PPO clipped surrogate: `max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))`
  - KL penalty: Schulman k3 `(r - log(r) - 1)` against reference

### 5. Configuration (`config.py`)

```python
FLOW_GRPO_REFUSION_CONFIG = {
    "group_size": 2,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "kl_coeff": 0.04,
    "clip_range": 0.2,
    "adv_clip_max": 5.0,
    "bf16": True,
    "logging_steps": 5,
    "grad_clip": 1.0,
    "num_generation_steps": 10,
    "generation_temperature": 0.7,
    "formfactory_port": 5050,
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}
```

### 6. Anyscale Job Configs

- `refusion_flow_grpo_job.yaml`: g5.2xlarge (24GB VRAM), 25 samples, 1 epoch
- `eval_refusion_flow_grpo_job.yaml`: Evaluation with 25 samples

## Key Differences from FS-DFM Flow-GRPO

| Aspect | FS-DFM Flow-GRPO | ReFusion Flow-GRPO |
|--------|------------------|-------------------|
| Log-prob | Poisson jump categorical (stay/jump) | Softmax at unmasked positions |
| Model loading | Custom DiT + manual LoRA | QLoRA via BitsAndBytes + PEFT |
| Forward pass | `model(x_t, t_emb)` -> logits | `model(input_ids, attention_mask)` -> logits |
| Mask handling | Uniform noise source distribution | Explicit mask token (ID 151670) |
| Reference model | Same DiT with frozen LoRA | Same ReFusion with frozen PEFT |
| VRAM | ~14GB (1.3B model) | ~20GB (8B quantized) |

## VRAM Budget (g5.2xlarge, 24GB)

- Policy model (8B, 4-bit quantized): ~5GB
- LoRA adapters + optimizer states: ~1GB
- Reference model (8B, 4-bit quantized, frozen): ~5GB
- Activations per forward pass (seq_len=512): ~3-5GB
- Gradient checkpointing overhead: ~1-2GB
- Buffer: ~5-6GB
- Total: ~20-23GB -- fits on g5.2xlarge

## Success Criteria

- Flow-GRPO rewards should match or exceed the existing GRPO trainer baseline
- Per-step log-probs should be finite and well-behaved (no NaN/Inf)
- Training should complete on g5.2xlarge without OOM
- Compare against baselines:
  - ReFusion SFT (existing)
  - ReFusion old GRPO (existing)
  - Target: Flow-GRPO >= old GRPO
