# ESPO: Sequence-Level ELBO Policy Optimization for DFM Web Agents

**Date**: 2026-02-19
**Branch**: `feat/flow-grpo`
**Status**: Design approved, implementing

## Problem

Seventeen iterations of token-level Flow-GRPO (v1-v18) produced 14/16 degradation
in held-out evaluation vs SFT. The token-level ELBO decomposition is fundamentally
unstable for discrete diffusion, as independently demonstrated by ESPO (arXiv:2512.03759),
CJ-GRPO, and MDPO.

## Solution

Replace token-level per-step REINFORCE with ESPO's sequence-level ELBO importance
ratios. Instead of computing per-step log-probs along the generation trajectory,
compute the full-sequence ELBO by randomly re-masking the completed output and
measuring cross-entropy. Use the ELBO as a single importance ratio for the entire
sequence.

## Architecture

### ReFusion ESPO (masked diffusion -- direct applicability)

ELBO computation:
1. Sample random mask count l from {1, ..., L_response}
2. Mask l random response positions with mask_token_id=151670
3. Forward pass, compute cross-entropy at masked positions
4. Weight by L/l (importance weighting for uniform l sampling)
5. Optional: coupled perturbation (mask complement) for variance reduction

Importance ratio: rho = exp((ELBO_theta - ELBO_old) / L)

### FS-DFM ESPO (GKL-based ELBO proxy)

ELBO computation:
1. Sample random timestep t ~ U(0,1)
2. Noise clean output via MixtureDiscreteProbPath forward process
3. Compute GKL loss at noised positions (standard FS-DFM training loss)
4. Negate to get ELBO (GKL loss = -ELBO inner term)
5. Optional: coupled perturbation (complementary mask)

Same importance ratio pattern as ReFusion.

### Training Loop

```
Phase 1: Generate G=4 rollouts (no trajectory needed)
Phase 2: Execute in browser, compute rewards
Phase 3: Compute GRPO advantages (group-relative)
Phase 4: Cache old_elbo and ref_elbo (no_grad, M=2 MC samples)
Phase 5: Policy update (mu iterations):
  - Recompute elbo_theta (with gradients, M=2 MC samples)
  - rho = exp((elbo_theta - old_elbo) / L)
  - clipped_rho = clamp(rho, 1-eps, 1+eps)
  - loss = -min(rho * A, clipped_rho * A).sum() / batch_size
  - kl = beta * (1/2) * (elbo_theta - ref_elbo)^2 / (batch_size * L)
  - loss += kl
  - backward + optimizer step
Phase 6: Early stopping, checkpointing (v13 mitigations)
```

### Key Hyperparameters

| Param | Value | Source |
|-------|-------|--------|
| mu (policy updates) | 1 | Conservative; v18 showed mu>1 degrades |
| MC samples (M) | 2 | ESPO paper default |
| Coupled perturbation | True | Halves ELBO variance |
| KL estimator | k2 | (1/2)(log_ratio)^2; ESPO proves k1/k3 fail |
| Beta (KL weight) | 3e-3 | ESPO Countdown recipe |
| Epsilon low | 0.2 | Standard PPO |
| Epsilon high | 0.2 | Standard PPO |
| Learning rate | 1e-5 | Match v18 |
| Group size G | 4 | Match v13-v18 |
| Gen steps T | 64 | Full denoising |
| Gen temp | 1.0 | Match v18 |
| Max steps | 40 | Early stopping |
| Zero-reward skip | min 1 | v13 mitigation |
| Dataset shuffle | True | v13 mitigation |

### File Structure

```
infra/training/flow_matching/
  espo_refusion_trainer.py
  espo_fsdfm_trainer.py
  config.py  (+ ESPO_REFUSION_CONFIG, ESPO_FSDFM_CONFIG)

infra/training/anyscale/
  espo_refusion_job.yaml
  espo_fsdfm_job.yaml
  eval_espo_refusion_job.yaml
  eval_espo_fsdfm_job.yaml
  submit_job.py  (+ 4 entries)
```

## Key Differences from Token-Level Flow-GRPO

| Aspect | Token-Level (v1-v18) | ESPO (v19) |
|--------|---------------------|------------|
| Action space | Per-step | Entire sequence |
| Log-prob | Per-step trajectory log-prob | ELBO over full sequence |
| Trajectory dependence | Requires stored trajectory | Only needs final sequence |
| Forward passes per rollout | T (one per step) | M*2 (MC samples, coupled) |
| KL estimator | k3 (Schulman) | k2 (quadratic) |
| Clipping | On per-step ratios | On sequence-level ratio |
| Policy updates per batch | 1 | mu (configurable) |

## References

- ESPO: arXiv:2512.03759 (Li et al., 2025)
- Code: https://github.com/ML-GSAI/ESPO
- CJ-GRPO: arXiv:2509.23924
- MDPO: arXiv:2508.13148
