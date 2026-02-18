# Training Pipeline - Session Summary & TODO

Last updated: 2026-02-18 03:05:42

---

## Current Session: Final Results + Paper Update + Full Training Config Fix

**Date**: 2026-02-18 03:05:42
**Branch**: `feat/flow-grpo`
**Duration context**: Collected final eval-grpo results. Updated STAD80 paper with clean n=124 results. Fixed all 10 training YAML configs to use full 992 train split.

### What Was Accomplished

- [x] Collected final eval-grpo results: Qwen3-8B GRPO 124/124 (100.0%), avg_reward=0.4551
- [x] Updated STAD80 project proposal with clean n=124 results (replaced old contaminated n=25 numbers)
- [x] Fixed all 10 training YAML configs: MAX_TRAIN_SAMPLES 25-500 -> 992 (full train split)
- [x] Updated results tables in docs/todo.md and infra/training/todo.md

### Key Findings

- **All training used artificially small subsets**: SFT used 200-500 of 992, GRPO used only 25 of 992. Train/val/test split already handles data separation. All configs now use full 992.
- **Qwen3-8B GRPO (100%/0.455) slightly worse than SFT (98.4%/0.486)**: Likely because GRPO only trained on 25 samples vs SFT's 200.

### Eval Results Table (n=124, val split, greedy -- trained on partial data)

| Model | Phase | Nonzero Rate | Avg Reward | Job ID | Note |
|-------|-------|-------------|------------|--------|------|
| Qwen3-8B SFT | SFT (clean) | 122/124 (98.4%) | 0.4857 | `prodjob_n2f7fv6gk2nb1bbikq51fwdmpe` | 200/992 train |
| Qwen3-8B GRPO | AR GRPO (clean) | 124/124 (100.0%) | 0.4551 | `prodjob_ssgpc8cipz9lu66fdlagd49mey` | 25/992 train |
| FS-DFM SFT | SFT (clean) | 53/124 (42.7%) | 0.1089 | `prodjob_rxumxvhsrywi1p3i3zcl4svgae` | 500/992 train |
| ReFusion SFT | SFT (clean) | 9/124 (7.3%) | 0.0177 | `prodjob_srlzyu5yxxf4pfa2jikrlv1v6v` | 200/992 train |
| FS-DFM old GRPO | old AR GRPO | 59/124 (47.6%) | 0.1032 | `prodjob_cyt2qim2dbcz2dhx6vjgx4l967` | Contaminated |
| ReFusion old GRPO | old AR GRPO | 4/124 (3.2%) | 0.0108 | `prodjob_q49gaad8sdfez93sbw41lvcxjp` | Contaminated |
| FS-DFM Flow-GRPO v6 | Flow-GRPO | 0/124 (0.0%) | 0.0000 | `prodjob_bh83drtmnxfm4y2patqfnrrmch` | Contaminated SFT |
| ReFusion Flow-GRPO v6 | Flow-GRPO | 26/124 (21.0%) | 0.0572 | `prodjob_8cjezz7l1ladk8tinij8guxf25` | Contaminated SFT |

### Next Steps

#### Immediate (next session)
- [ ] Commit YAML config changes (MAX_TRAIN_SAMPLES=992) and STAD80 paper update
- [ ] Re-train all 3 SFT models on full 992 samples: finetuning-sft, flow-llm-sft, fsdfm-sft
- [ ] After SFT completes, re-train GRPO from new full-data SFT checkpoints
- [ ] Run eval on val split for all new checkpoints

#### Short-term
- [ ] Create PR from `feat/flow-grpo` to `main`
- [ ] Run test split evaluation for final paper numbers (EVAL_SPLIT=test, n=124)
- [ ] Update STAD80 paper with full-data results

#### Future
- [ ] FS-DFM GRPO: investigate ReinFlow-inspired learnable noise/temperature schedules
- [ ] ReFusion Flow-GRPO: reduce kl_coeff (v6's kl=0.0003 beat v8's kl=0.2355)
- [ ] KL coefficient sweep
- [ ] Align training output format with AgentOutput JSON for framework integration

---

## Previous Session: Job Submission -- SFT Re-training + Full Val Eval (n=124)

**Date**: 2026-02-17 20:24:03
**Branch**: `feat/flow-grpo`
**Duration context**: Implemented the stratified train/val/test split plan (7 commits from `local_docs/plans/train-val-test-split.md`). Analyzed framework output format vs training format (AgentOutput JSON vs plain text steps). Increased eval from n=25 to n=124 (full val set). Added seed support to all trainers and eval scripts.

### What Was Accomplished

- [x] **Implemented train/val/test split** (9 commits):
  1. Added `split_dataset()` to `formfactory_preprocessor.py` -- stratified by form_name, seed=42, 80/10/10
  2. Updated `DATA_CONFIG` in both `flow_matching/config.py` and `finetuning/config.py` -- train_file, val_file, test_file with env var overrides
  3. Updated 3 eval scripts (`eval_refusion_sft.py`, `eval_fsdfm_sft.py`, `eval_sft.py`) to load from val/test split via `EVAL_SPLIT` env var
  4. Updated `finetuning/sft_trainer.py` to load from pre-split train and val files (removed `train_test_split()`)
  5. Updated 10 training Anyscale YAML files to point at `formfactory_sft_train.jsonl`/`formfactory_flow_train.jsonl`
  6. Updated 8 eval Anyscale YAML files to point at `formfactory_sft_val.jsonl` with `FLOW_VAL_FILE`/`VAL_FILE` env vars
  7. Generated split files: 992 train / 124 val / 124 test. Verified zero overlap, all 25 form types in each split.
  8. Increased `MAX_EVAL_SAMPLES` from 25 to 124 in all 8 eval YAML files
  9. Added `torch.manual_seed` seed support to finetuning SFT trainer and all 3 eval scripts (RANDOM_SEED env var, default 42)
- [x] **Analyzed framework output format** (AgentOutput):
  - Framework expects structured JSON via LLM tool/function calling: `AgentOutput` with `thinking`, `evaluation_previous_goal`, `memory`, `next_goal`, `action: list[ActionModel]`
  - Training uses plain text step-by-step instructions parsed by regex (`action_parser.py`)
  - These are two completely different formats -- fine-tuned models cannot plug into the main Agent class directly
  - Fine-tuned models are single-shot planners (no multi-turn browser state observation)

### Key Findings

- **Framework vs training format gap**: The OpenBrowserAI `AgentOutput` class (`src/openbrowser/agent/views.py:149`) expects JSON with action objects (`click_element`, `input_text`, `navigate`, etc.). The SFT training data produces plain text like `"Step 1: Type 'value' into the 'field' field"`. The `parse_rollout_to_actions()` regex bridge in `infra/training/shared/action_parser.py` converts between them, but this only works in the eval pipeline -- not in the main agent loop.
- **Train/eval data overlap eliminated**: Previously all trainers and eval scripts loaded from the same 1,240-example file. Now training uses 992 examples, eval uses a separate 124-example val set.
- **Eval variance reduced**: Moving from n=25 to n=124 reduces standard error by ~2.2x (sqrt(124/25)). This makes results more reliable for comparing models.
- **Subagent model routing**: `model: "opus"` in Task tool routes to Opus 4.1, not 4.6. Omitting the model parameter inherits parent model (Opus 4.6). Added global Claude rule for this.

### Files Modified

| File | Change |
|------|--------|
| `infra/training/shared/formfactory_preprocessor.py` | Added `split_dataset()`, updated `preprocess()` to generate 6 split files |
| `infra/training/flow_matching/config.py` | DATA_CONFIG: added val_file, test_file; removed eval_split |
| `infra/training/finetuning/config.py` | DATA_CONFIG: added val_file, test_file; removed eval_split |
| `infra/training/flow_matching/eval_refusion_sft.py` | Load from val/test split, added seed support |
| `infra/training/flow_matching/eval_fsdfm_sft.py` | Load from val/test split, added seed support |
| `infra/training/finetuning/eval_sft.py` | Load from val/test split, added seed support |
| `infra/training/finetuning/sft_trainer.py` | Load pre-split files instead of train_test_split(), added seed |
| 10 training YAML files | TRAIN_FILE -> train split files |
| 8 eval YAML files | TRAIN_FILE -> VAL_FILE, MAX_EVAL_SAMPLES 25->124 |

### Important Commands

```bash
# Run preprocessor to regenerate split files
uv run python -m infra.training.shared.formfactory_preprocessor

# Verify split sizes and no overlap
python3 -c "
import json
from collections import Counter
for name in ['train', 'val', 'test']:
    path = f'data/processed/formfactory_sft_{name}.jsonl'
    with open(path) as f:
        data = [json.loads(l) for l in f]
    forms = Counter(d['form_name'] for d in data)
    print(f'{name}: {len(data)} examples, {len(forms)} form types')
"

# Submit eval with full val set (124 prompts)
uv run python -m infra.training.anyscale.submit_job eval-refusion-sft
uv run python -m infra.training.anyscale.submit_job eval-refusion-flow-grpo
```

### Important Code

**split_dataset()** (`infra/training/shared/formfactory_preprocessor.py:174-210`):

```python
def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in data:
        groups[item.get("form_name", "unknown")].append(item)
    train, val, test = [], [], []
    for form_name in sorted(groups.keys()):
        items = groups[form_name]
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test
```

**AgentOutput schema** (`src/openbrowser/agent/views.py:149-159`):

```python
class AgentOutput(BaseModel):
    thinking: str | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action: list[ActionModel]  # structured action objects
```

### Important Specs & Contracts

**Framework expected format (AgentOutput JSON via tool use)**:

```json
{
  "evaluation_previous_goal": "Success - navigated to form page",
  "memory": "On exhibition submission form page",
  "next_goal": "Fill in Artist Name field",
  "action": [{"input_text": {"index": 3, "text": "Amanda Lee", "clear": true}}]
}
```

**Training format (plain text, parsed by regex)**:

```
Step 1: Navigate to http://127.0.0.1:5050/arts-creative/exhibition-submission
Step 2: Click on the 'Artist Name' input field
Step 3: Type 'Amanda Lee' into the 'Artist Name' field
...
Step 19: Click the Submit button
```

**Dataset split sizes**: train=992, val=124, test=124 (seed=42, stratified by form_name)

### References

- Split plan: `local_docs/plans/train-val-test-split.md`
- AgentOutput class: `src/openbrowser/agent/views.py:149`
- Action parser (regex bridge): `infra/training/shared/action_parser.py`
- Tool action models: `src/openbrowser/tools/views.py`

### Current State

**Git**: on `feat/flow-grpo`, 48 commits ahead of `main`. Modified: `infra/training/todo.md`. Untracked: `data/processed/`, `extension.zip`.

**Generated data files** (in `data/processed/`, gitignored):
- `formfactory_sft.jsonl` (1,240 -- unsplit, backward compat)
- `formfactory_sft_train.jsonl` (992)
- `formfactory_sft_val.jsonl` (124)
- `formfactory_sft_test.jsonl` (124)
- `formfactory_flow.jsonl`, `formfactory_flow_train.jsonl`, `formfactory_flow_val.jsonl`, `formfactory_flow_test.jsonl` (same splits)

### Next Steps

#### Immediate (next session)
- [ ] Push split commits to remote and re-run SFT training with proper train split (992 examples instead of full 1,240)
- [ ] Submit eval jobs for all models on val split (n=124): ReFusion SFT, FS-DFM SFT, ReFusion Flow-GRPO v6, ReFusion Flow-GRPO v8
- [ ] Re-evaluate SFT baselines with seed=42 for reproducible comparison

#### Short-term
- [x] Implement train/val/test split (done this session)
- [x] Increase eval from n=25 to n=124 (done this session)
- [x] Add seeds to all trainers and eval scripts (done this session)
- [ ] Create PR from `feat/flow-grpo` to `main`
- [ ] Run test split evaluation for final paper numbers (EVAL_SPLIT=test, n=124)

#### Future
- [ ] ReFusion Flow-GRPO v6 is the winner -- v8's higher KL hurt greedy eval. Consider reducing lr or kl_coeff for v9.
- [ ] FS-DFM GRPO: investigate SDE-based log-probs (continuous Flow-GRPO) for Poisson jump process
- [ ] ReFusion: try larger training budget (50+ prompts, multiple epochs) with train split
- [ ] KL coefficient sweep: v6 (kl=0.0003) beat v8 (kl=0.2355) -- optimal KL is much lower than expected
- [ ] Implement proper PPO with K>1 optimization epochs
- [ ] Consider top-k or nucleus sampling instead of temperature
- [ ] Align training output format with AgentOutput JSON for framework integration (or build adapter)

---

## Previous Sessions

### 2026-02-17 18:40 -- v8 Results Collection + Old GRPO Eval + SFT Discrepancy Investigation

Collected v8 training results (FS-DFM SUCCEEDED 3/100 still broken, ReFusion SUCCEEDED 74/100 with kl=0.2355 -- real learning). Collected old GRPO eval results under new metric (catastrophic: old ReFusion GRPO 2/25 (8%), old FS-DFM GRPO 13/25 (52%)). Investigated SFT discrepancy vs proposal: caused by metric change from binary exact-match to continuous string similarity, not model regression. Submitted v8 eval jobs for both models. v8 eval results: FS-DFM 0/25 (0%), ReFusion 13/25 (52%)/0.126 -- v8 worse than v6 despite higher KL.

### Eval Results Table (all under new metric with field_accuracy, greedy, n=25)

| Model | Phase | Nonzero Rate | Avg Reward | Source |
|-------|-------|-------------|------------|--------|
| FS-DFM SFT | SFT | 18/25 (72%) | 0.1681 | `prodjob_j2y6ekhujseese6qdwmuj9caqm` |
| ReFusion SFT | SFT | 13/25 (52%) | 0.2074 | `prodjob_7fuzb7p9g83qr7fzsrhrzly9d3` |
| FS-DFM old GRPO | old GRPO | 13/25 (52%) | 0.1272 | `prodjob_fjbphjsic2114y6r7vgpdwbr6d` |
| ReFusion old GRPO | old GRPO | 2/25 (8%) | 0.0284 | `prodjob_vmsczeyw1lxl8izdiz3x7w1v1c` |
| FS-DFM Flow-GRPO v6 | Flow-GRPO | 0/25 (0%) | 0.0000 | `prodjob_epw4isnzgrtuwqb9ui4rg7nm92` |
| ReFusion Flow-GRPO v6 | Flow-GRPO | 15/25 (60%) | 0.2537 | `prodjob_im2us8lg2xq67m4v9vp8454s4e` |
| FS-DFM Flow-GRPO v8 | Flow-GRPO | 0/25 (0%) | 0.0000 | `prodjob_mpc7h5kuw8vrb4ve6b9nii241f` |
| ReFusion Flow-GRPO v8 | Flow-GRPO | 13/25 (52%) | 0.1260 | `prodjob_mvudwc9s5uebbqlipav1bmvigg` |

### Flow-GRPO Version History (complete through v8)

| Ver | FS-DFM Temp | FS-DFM Train | FS-DFM Eval | ReFusion Temp | ReFusion Train | ReFusion Eval | Fix Applied |
|-----|-------------|--------------|-------------|---------------|----------------|---------------|-------------|
| v1 | 10/1.0 | TERMINATED (NaN) | -- | 10/0.7 | 3/50 (6%) | -- | -- |
| v2 | 32/0.7 | 5/50 (10%), 0.020 | -- | 20/0.7 | 18/50 (36%), 0.072 | -- | NaN guards |
| v3 | 64/0.7 | 1/50 (2%), 0.004 | -- | 64/0.7 | 42/50 (84%), 0.168 | 76%/0.152 | T=64, special_tokens |
| v4 | 64/0.7 | 9/50 (18%), kl=0 | 0%/0.000 | 64/0.7 | 42/50 (84%), kl=0 | 52%/0.120 | Temp mismatch fix |
| v5 | 64/0.3 | 7/100 (7%), kl=0 | 0%/0.000 | 64/0.7 | 87/100 (87%), kl=0 | 60%/0.216* | G=4, FS-DFM temp=0.3 |
| v6 | 64/0.7 | 4/100 (4%), kl=0 | 0%/0.000 | 64/0.7 | 88/100 (88%), kl=0.0003 | 60%/0.254 | Reward fix (field_accuracy) |
| v7 | 64/1.0 | 0/10 (0%), kl=0 | -- | 64/1.0 | 81/100 (81%), kl=0.015 | pending | temp=1.0, confidence noise |
| v8 | 64/1.0 | 3/100 (3%), kl=0 | 0%/0.000 | 64/1.0 | 74/100 (74%), kl=0.236 | 52%/0.126 | Norm fix, K=8, lr=5e-5 |

### Config Values (current, after v8 fixes)

- `FLOW_GRPO_FSDFM_CONFIG`: G=4, lr=5e-5, kl_coeff=0.04, T=64, temp=1.0, num_sampled_timesteps=8
- `FLOW_GRPO_REFUSION_CONFIG`: G=4, lr=5e-5, kl_coeff=0.04, T=64, temp=1.0, confidence_noise_std=0.1, num_sampled_timesteps=8

### 2026-02-17 18:13 -- Flow-GRPO v8 (Proper Implementation) + v7 Results + Old GRPO Reproduction

Investigated v7 results: FS-DFM 0/10 nonzero (ZERO rewards despite temp=1.0), ReFusion step 15/25 with avg_reward=0.305 kl=0.0117 (first nonzero KL). Root-caused FS-DFM failure to loss normalization bug: dividing by valid_terms=T*G=256 instead of per-rollout K, making effective lr ~4e-8. Also found: no denoising reduction (processing all 64 steps instead of K=8), redundant GPU transfers. Applied 6 commits fixing both trainers per Flow-GRPO paper (Liu et al. 2025). Re-ran old GRPO trainers to reproduce proposal baselines. Collected eval results under new metric.

### 2026-02-17 17:24 -- Flow-GRPO v6 Results + v7 Fixes + Baseline Re-evals + .tex Reconstruction

Checked v6 training results -- FS-DFM still generating garbage (4/100 at temp=0.7), ReFusion getting continuous rewards (0.422) but kl=0 due to near-identical within-group outputs. Root-caused both: FS-DFM needs temp=1.0 (old GRPO baseline), ReFusion position selection is deterministic (confidence-based top-k). Applied fixes: temp=1.0 for both, added confidence_noise_std=0.1 for ReFusion position diversity, added torch.manual_seed to both SFT trainers. Submitted v7 training jobs. Submitted 5 eval jobs (v6 + 4 baseline re-evals under new metric). Reconstructed lost STAD80_project_proposal.tex from PDF.

### 2026-02-17 16:48 -- Flow-GRPO v5 Results + Reward Function Fix + v6 Submission

Checked v5 results (both kl=0 confirming binary reward problem). Root-caused the dead field_accuracy reward: BrowserOutcome.submitted_values was never populated, making 40% of reward always 0. Fixed by tracking filled field values during browser action execution. Reverted FS-DFM temp from 0.3 back to 0.7. Submitted v6 training jobs with reward fix. Early v6 results: ReFusion reward=0.422 (up from 0.200), FS-DFM slight variance (0.200, 0.217). Investigated SFT regression vs proposal -- confirmed random variance from n=25 + no seeds. Removed local_docs/ from git history with filter-repo.

### 2026-02-17 16:22 -- Flow-GRPO v4 Results + v5 Submission + Git History Cleanup

Checked v4 results (both kl=0, G=2 too small). Investigated SFT performance regression (n=25 variance). Applied fixes: G=4 for both models, temp=0.3 for FS-DFM. Submitted v5 training jobs. v4 eval confirmed: ReFusion 52%/0.120 (matches SFT), FS-DFM 0%/0.000 (broken under greedy). v5 early results at Step 5 still showed kl=0 even with G=4 -- predicted binary reward root cause. Cleaned git history (removed STAD80/CSC490 with filter-repo).

### 2026-02-17 15:03 -- Flow-GRPO v2/v3/v4 Debugging + Temperature Mismatch Fix

Checked v2 results (both still garbage), iterated through v3 and v4 fixes. Found two root causes: (1) denoising steps still too low at T=32/T=20, need T=64 to match eval; (2) temperature mismatch between generation (temp=0.7) and log-prob computation (implicit temp=1.0) caused wrong gradients. ReFusion v3 at T=64 achieved 84% nonzero training (best result yet). FS-DFM v3 worse than v2 despite T=64 due to temperature mismatch destroying weights. Fixed temperature, submitted v4.

### 2026-02-17 14:00 -- SFT Completion + Flow-GRPO Diagnosis + Denoising Fix

Checked SFT job results (both SUCCEEDED), submitted SFT evals and Flow-GRPO jobs. Discovered Flow-GRPO produced garbage text (FS-DFM: `!!!!!`, ReFusion: garbled repetitions) despite SFT checkpoints loading correctly. Root-caused to T=10 denoising steps being too coarse (eval uses T=64). Fixed configs to T=32/T=20, added NaN guards, resubmitted.

### 2026-02-17 13:23 -- NaN Fix + SFT Prerequisite Discovery + Job Re-submission

Checked both Flow-GRPO job results from previous session. FS-DFM FAILED with NaN/Inf crash in discrete Euler solver (bf16 softmax overflow). ReFusion "SUCCEEDED" but with 0/50 nonzero rewards across all 25 prompts. Root cause for both: SFT checkpoints missing on Anyscale storage, so models generated garbage text. Fixed NaN numerical stability, added debug logging, terminated broken jobs, submitted SFT training as prerequisite.

### 2026-02-17 12:56 -- CUDA OOM Fix + REINFORCE Simplification + Job Re-submission

Diagnosed and fixed CUDA OOM in FS-DFM Flow-GRPO trainer. Found critical bug: old_log_prob was identical to log_prob (ratio always 1.0, PPO clipping was a no-op). Simplified to REINFORCE, added per-step gradient accumulation, moved ref_model to CPU. Applied same fixes to ReFusion trainer. Terminated stale ReFusion job (0% reward after 15 prompts). Re-submitted both jobs with fixes.

### 2026-02-17 12:39 -- FormFactory Packaging Fix

Fixed `data/formfactory/` not being packaged in Anyscale jobs (`.gitignore` excluded it from Ray's `working_dir`). Added `git` to 3 Containerfiles, updated 14 YAML entrypoints to download FormFactory at runtime via `download_datasets.py`.

### 2026-02-17 12:16 -- FormFactory Bug Discovery

Discovered FS-DFM job reported `SUCCESS` but actually failed at runtime: `FormFactory app.py not found`. Root cause: `.gitignore` line `data/formfactory/` prevents Ray from uploading the directory.

### 2026-02-17 12:05 -- Flow-GRPO Implementation for FS-DFM + ReFusion 8B

Committed 5 FS-DFM Flow-GRPO commits (trajectory-recording Euler solver, trainer, config, Anyscale jobs). Submitted FS-DFM training job. Designed and implemented full ReFusion Flow-GRPO pipeline (7 commits): `UnmaskingTrajectory` dataclasses, `generate_with_trajectory()`, `compute_unmasking_step_log_prob()`, trainer, configs.

### 2026-02-17 11:13 -- Initial Flow-GRPO for FS-DFM

Implemented Flow-GRPO (Liu et al., 2025) for FS-DFM 1.3B. Discrete analog of continuous Flow-GRPO: per-position Poisson jump gives categorical distribution, log-prob computed exactly for stay/jump transitions. Trajectory recording stores (x_t, x_next) at each Euler step (~160KB). Created trainer, job configs, registered in submit_job.py.

### Key References (all sessions)

- Design doc: `docs/plans/2026-02-17-refusion-flow-grpo-design.md`
- Implementation plan: `docs/plans/2026-02-17-refusion-flow-grpo-plan.md`
- Flow-GRPO paper: arxiv.org/abs/2505.05470 (Liu et al., 2025); code: github.com/yifan123/flow_grpo
- Config: `FLOW_GRPO_REFUSION_CONFIG` -- G=4, lr=5e-5, kl_coeff=0.04, T=64, temp=1.0, confidence_noise_std=0.1, K=8
- Config: `FLOW_GRPO_FSDFM_CONFIG` -- G=4, lr=5e-5, kl_coeff=0.04, T=64, temp=1.0, K=8
- Split plan: `local_docs/plans/train-val-test-split.md`
