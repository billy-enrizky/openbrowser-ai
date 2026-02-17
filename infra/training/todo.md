# Training Pipeline - Session Summary & TODO

## Session: 2026-02-17 11:13

### What Was Done

Implemented Flow-GRPO (Liu et al., 2025) for FS-DFM 1.3B discrete flow matching model. This adapts the continuous Flow-GRPO algorithm (designed for image diffusion models) to the discrete Poisson jump process used by FS-DFM for text generation.

**Problem solved**: The existing `fsdfm_online_grpo_trainer.py` used advantage-weighted SFT loss (`advantage * generalized_kl_loss`), which is NOT a proper policy gradient. It degraded FS-DFM performance by -29% vs SFT. Flow-GRPO computes proper per-step log-probabilities aligned with the actual generation process (Poisson jump CTMC), enabling PPO-style clipped policy gradients.

**Files created:**
- `infra/training/flow_matching/fsdfm_flow_grpo_trainer.py` -- Main Flow-GRPO trainer
- `infra/training/anyscale/fsdfm_flow_grpo_job.yaml` -- Training job (25 samples, 1 epoch)
- `infra/training/anyscale/eval_fsdfm_flow_grpo_job.yaml` -- Eval job (25 samples)

**Files modified:**
- `infra/training/flow_matching/fsdfm_model.py` -- Added trajectory-recording solver, per-step log-prob computation
- `infra/training/flow_matching/config.py` -- Added `FLOW_GRPO_FSDFM_CONFIG`
- `infra/training/anyscale/submit_job.py` -- Registered `fsdfm-flow-grpo` and `eval-fsdfm-flow-grpo` jobs
- `local_docs/CHANGELOG.md` -- Updated

### Key Technical Details

- **Discrete analog of continuous Flow-GRPO**: Per-position Poisson jump gives categorical distribution; log-prob computed exactly for stay/jump transitions
- **Trajectory recording**: Stores (x_t, x_next) at each Euler step (~160KB per trajectory)
- **Denoising reduction**: T=10 generation steps (vs 64 for inference)
- **PPO-style clipping**: clip_range=0.2, iterates over ALL timesteps per trajectory
- **KL penalty**: Schulman k3 (r - log(r) - 1) against frozen SFT reference model
- **VRAM**: ~14GB on A10G (24GB), fits both policy + reference models

### Next Steps / TODO

- [ ] Submit SFT job to Anyscale (if fresh SFT needed): `uv run infra/training/anyscale/submit_job.py fsdfm-sft`
- [ ] Submit Flow-GRPO job to Anyscale: `uv run infra/training/anyscale/submit_job.py fsdfm-flow-grpo`
- [ ] After training completes, submit eval: `uv run infra/training/anyscale/submit_job.py eval-fsdfm-flow-grpo`
- [ ] Compare Flow-GRPO results with existing baselines:
  - FS-DFM SFT: 17/25 nonzero (68%), avg_reward=0.136
  - FS-DFM old GRPO: 12/25 nonzero (48%), avg_reward=0.096
  - Target: Flow-GRPO should match or exceed SFT performance
- [ ] If T=10 generation steps produce low-quality text (all zero rewards), increase to T=20
- [ ] Consider tuning kl_coeff (currently 0.04) and clip_range (currently 0.2)
- [ ] If results are positive, run full-dataset evaluation (1,240 prompts)
- [ ] Update STAD80 project proposal with Flow-GRPO results
- [x] Implement Flow-GRPO for ReFusion 8B (completed 2026-02-17 12:00)

### ReFusion Flow-GRPO -- Next Steps

- [ ] Submit FS-DFM Flow-GRPO job (submitted, image building): `uv run infra/training/anyscale/submit_job.py fsdfm-flow-grpo`
- [ ] Submit ReFusion Flow-GRPO job: `uv run infra/training/anyscale/submit_job.py refusion-flow-grpo`
- [ ] After FS-DFM training completes, submit eval: `uv run infra/training/anyscale/submit_job.py eval-fsdfm-flow-grpo`
- [ ] After ReFusion training completes, submit eval: `uv run infra/training/anyscale/submit_job.py eval-refusion-flow-grpo`
- [ ] Compare ReFusion Flow-GRPO results with existing baselines:
  - ReFusion SFT (existing)
  - ReFusion old GRPO (existing, AR log-probs)
  - Target: Flow-GRPO >= old GRPO
- [ ] If T=10 denoising steps produce low quality, increase to T=20
- [ ] If results are positive, run full-dataset evaluation (1,240 prompts)
- [ ] Update STAD80 project proposal with Flow-GRPO results for both models

### Commit Plan (completed)

All commits on `feat/flow-grpo` branch:
1. [x] `feat(training): add trajectory-recording Euler solver and per-step log-prob to fsdfm_model.py`
2. [x] `feat(training): add FLOW_GRPO_FSDFM_CONFIG to flow matching config`
3. [x] `feat(training): implement Flow-GRPO trainer for FS-DFM with discrete policy gradients`
4. [x] `feat(infra): add Anyscale job configs for Flow-GRPO training and evaluation`
5. [x] `docs: add training session summary and TODO for Flow-GRPO next steps`
6. [x] `docs: add ReFusion Flow-GRPO design document`
7. [x] `feat(training): add UnmaskingTrajectory data structures for ReFusion Flow-GRPO`
8. [x] `feat(training): add generate_with_trajectory() to FlowLLM for ReFusion Flow-GRPO`
9. [x] `feat(training): add compute_unmasking_step_log_prob() for ReFusion trajectory log-probs`
10. [x] `feat(training): add FLOW_GRPO_REFUSION_CONFIG for ReFusion 8B Flow-GRPO`
11. [x] `feat(training): implement ReFusion Flow-GRPO trainer with masked diffusion policy gradients`
12. [x] `feat(infra): add Anyscale job configs for ReFusion Flow-GRPO training and evaluation`
