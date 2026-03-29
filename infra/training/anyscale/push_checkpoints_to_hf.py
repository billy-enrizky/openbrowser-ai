"""Push trained checkpoints from Anyscale persistent storage to HuggingFace Hub.

Reads model checkpoints from /mnt/user_storage/openbrowser/checkpoints/
and uploads them as separate HuggingFace repos.

Usage (on Anyscale):
    HF_TOKEN=... python -m infra.training.anyscale.push_checkpoints_to_hf

Models pushed:
    1. billyenrizky/FlowVFE-39M-SFT           (flow-sft)
    2. billyenrizky/FlowVFE-39M-FlowGRPO       (online-flow-grpo)
    3. billyenrizky/FS-DFM-1.3B-SFT            (fsdfm-sft)
    4. billyenrizky/FS-DFM-1.3B-FlowGRPO       (online-fsdfm-grpo)
    5. billyenrizky/ReFusion-8B-SFT            (flow-llm-sft)
    6. billyenrizky/ReFusion-8B-ESPO-mu8       (espo-refusion)
    7. billyenrizky/ReFusion-8B-CJ-GRPO       (cjgrpo-refusion)
    8. billyenrizky/ReFusion-8B-MDPO           (mdpo-refusion)
    9. billyenrizky/FS-DFM-1.3B-ESPO-mu8      (espo-fsdfm)
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STORAGE_BASE = Path("/mnt/user_storage/openbrowser/checkpoints")
HF_ORG = "billyenrizky"

# (checkpoint_dir_name, hf_repo_suffix, model_card_description)
MODELS = [
    (
        "flow-sft",
        "FlowVFE-39M-SFT",
        "39M-parameter FlowVFE model fine-tuned with SFT on FormFactory web form-filling tasks. "
        "Trained from scratch with byte-level tokenization and discrete flow matching. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "online-flow-grpo",
        "FlowVFE-39M-FlowGRPO",
        "39M-parameter FlowVFE model trained with Flow-GRPO (online RL with browser execution) "
        "on FormFactory web form-filling tasks. Built on the FlowVFE-39M-SFT checkpoint. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "fsdfm-sft",
        "FS-DFM-1.3B-SFT",
        "FS-DFM 1.3B (Apple) fine-tuned with SFT on FormFactory web form-filling tasks. "
        "Uses LoRA adapters on the DiT architecture with Poisson jump sampling. "
        "Achieves 68.5% nonzero reward rate and 0.146 average reward on 124 test tasks. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "online-fsdfm-grpo",
        "FS-DFM-1.3B-FlowGRPO",
        "FS-DFM 1.3B trained with Flow-GRPO v13 DCE (Denoising Cross Entropy) on FormFactory. "
        "Best token-level RL config for FS-DFM: 73.4% nonzero rate / 0.159 average reward "
        "on 124 test tasks (+4.9pp / +8.9% vs SFT). "
        "Uses advantage-weighted GKL loss aligned with the diffusion denoising trajectory. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "flow-llm-sft",
        "ReFusion-8B-SFT",
        "ReFusion 8B (GSAI-ML) fine-tuned with QLoRA SFT on FormFactory web form-filling tasks. "
        "Uses masked diffusion with iterative unmasking on Qwen3-8B backbone. "
        "Achieves 60.5% nonzero reward rate and 0.267 average reward on 124 test tasks. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "espo-refusion",
        "ReFusion-8B-ESPO-mu8",
        "ReFusion 8B trained with ESPO mu=8 (ELBO-based Sequence-level Policy Optimization). "
        "Achieves 83.1% nonzero rate / 0.394 average reward on 124 test tasks (+22.6pp over SFT). "
        "Sequence-level RL with multi-epoch training (mu=8) and PPO clipping. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "cjgrpo-refusion",
        "ReFusion-8B-CJ-GRPO",
        "ReFusion 8B trained with CJ-GRPO (Consistency-Justified GRPO). "
        "Achieves 83.9% nonzero rate / 0.390 average reward on 124 test tasks (+23.4pp over SFT). "
        "Per-step trajectory consistency with mu=1. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "mdpo-refusion",
        "ReFusion-8B-MDPO",
        "ReFusion 8B trained with MDPO (Masked Diffusion Policy Optimization). "
        "Best result in the paper: 91.9% nonzero rate / 0.445 average reward on 124 test tasks "
        "(+31.4pp over SFT). Temporal advantage decomposition with mu=1. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
    (
        "espo-fsdfm",
        "FS-DFM-1.3B-ESPO-mu8",
        "FS-DFM 1.3B trained with ESPO mu=8 (ELBO-based Sequence-level Policy Optimization). "
        "First RL method to improve FS-DFM over SFT: 87.1% nonzero rate / 0.198 average reward "
        "on 124 test tasks (+18.6pp over SFT). Only ELBO-based methods generalize to DFM architectures. "
        "Part of the STAD80 project: Generative Action Planning via Discrete Flow Matching.",
    ),
]


def create_model_card(repo_name: str, description: str) -> str:
    """Generate a HuggingFace model card."""
    return f"""---
tags:
  - discrete-flow-matching
  - web-action-planning
  - formfactory
  - reinforcement-learning
  - openbrowser
license: apache-2.0
---

# {repo_name}

{description}

## Paper

**Generative Action Planning via Discrete Flow Matching with Online Reinforcement Fine-Tuning**
- Author: Muhammad Enrizky Brillian
- Institution: University of Toronto Scarborough

## Training Details

- **Dataset**: FormFactory (992 train / 124 val / 124 test tasks, 25 form types, 8 domains)
- **Infrastructure**: Single NVIDIA A10G GPU (24GB VRAM) on Anyscale
- **Framework**: PyTorch + PEFT (LoRA/QLoRA)

## Citation

If you use this model, please cite:

```bibtex
@article{{brillian2026flowgrpo,
  title={{Generative Action Planning via Discrete Flow Matching with Online Reinforcement Fine-Tuning}},
  author={{Brillian, Muhammad Enrizky}},
  year={{2026}}
}}
```

This model was trained and evaluated on the FormFactory benchmark:

```bibtex
@misc{{li2025formfactory,
  title={{FormFactory: An Interactive Benchmarking Suite for Multimodal Form-Filling Agents}},
  author={{Bobo Li and Yuheng Wang and Hao Fei and Juncheng Li and Wei Ji and Mong-Li Lee and Wynne Hsu}},
  year={{2025}},
  eprint={{2506.01520}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}},
  url={{https://arxiv.org/abs/2506.01520}}
}}
```
"""


def push_model(checkpoint_name: str, repo_suffix: str, description: str):
    """Push a single checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi

    checkpoint_path = STORAGE_BASE / checkpoint_name
    repo_id = f"{HF_ORG}/{repo_suffix}"

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path} -- skipping {repo_id}")
        return False

    # Check it has files
    files = list(checkpoint_path.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    if file_count == 0:
        logger.warning(f"Checkpoint dir empty: {checkpoint_path} -- skipping {repo_id}")
        return False

    logger.info(f"Pushing {checkpoint_path} ({file_count} files) -> {repo_id}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set")
        return False

    api = HfApi(token=token)

    # Create repo if needed
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

    # Write model card
    model_card = create_model_card(repo_suffix, description)
    card_path = checkpoint_path / "README.md"
    card_path.write_text(model_card)

    # Upload entire directory
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        commit_message=f"Upload {repo_suffix} checkpoint",
    )

    logger.info(f"Pushed {repo_id} -- https://huggingface.co/{repo_id}")
    return True


def main():
    if not STORAGE_BASE.exists():
        logger.error(
            f"Storage base {STORAGE_BASE} does not exist. "
            "This script must run on an Anyscale node with persistent storage."
        )
        sys.exit(1)

    logger.info(f"Checking checkpoints in {STORAGE_BASE}")
    available = [d.name for d in STORAGE_BASE.iterdir() if d.is_dir()]
    logger.info(f"Available checkpoints: {available}")

    success_count = 0
    for checkpoint_name, repo_suffix, description in MODELS:
        try:
            if push_model(checkpoint_name, repo_suffix, description):
                success_count += 1
        except Exception:
            logger.exception(f"Failed to push {repo_suffix}")

    logger.info(f"Done: {success_count}/{len(MODELS)} models pushed to HuggingFace")
    if success_count < len(MODELS):
        logger.warning("Some models were not pushed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
