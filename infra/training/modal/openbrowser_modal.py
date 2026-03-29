"""
OpenBrowser Training Jobs -- Modal Deployment
==============================================

Run 6 RL training jobs (ESPO mu=8, CJ-GRPO, MDPO) on ReFusion 8B and
FS-DFM 1.3B discrete diffusion web agents via Modal.com.

Pre-requisites (one-time)
-------------------------
    modal secret create openbrowser-secrets HF_TOKEN=hf_xxx
    modal run infra/training/modal/openbrowser_modal.py::download_sft_checkpoints

Usage
-----
All 6 jobs:
    modal run infra/training/modal/openbrowser_modal.py

Individual training jobs:
    modal run infra/training/modal/openbrowser_modal.py::espo_refusion_mu8
    modal run infra/training/modal/openbrowser_modal.py::espo_fsdfm_mu8
    modal run infra/training/modal/openbrowser_modal.py::cjgrpo_refusion
    modal run infra/training/modal/openbrowser_modal.py::cjgrpo_fsdfm
    modal run infra/training/modal/openbrowser_modal.py::mdpo_refusion
    modal run infra/training/modal/openbrowser_modal.py::mdpo_fsdfm

mu=8 stabilization experiments (FS-DFM only):
    modal run --detach infra/training/modal/openbrowser_modal.py::cjgrpo_fsdfm_mu8
    modal run --detach infra/training/modal/openbrowser_modal.py::mdpo_fsdfm_mu8

All 12 eval jobs (6 checkpoints x val+test):
    modal run infra/training/modal/openbrowser_modal.py::eval_all

Individual eval jobs:
    modal run infra/training/modal/openbrowser_modal.py::eval_espo_refusion_mu8
    modal run infra/training/modal/openbrowser_modal.py::eval_espo_fsdfm_mu8
    modal run infra/training/modal/openbrowser_modal.py::eval_cjgrpo_refusion
    modal run infra/training/modal/openbrowser_modal.py::eval_cjgrpo_fsdfm
    modal run infra/training/modal/openbrowser_modal.py::eval_mdpo_refusion
    modal run infra/training/modal/openbrowser_modal.py::eval_mdpo_fsdfm
    modal run infra/training/modal/openbrowser_modal.py::eval_cjgrpo_fsdfm_mu8
    modal run infra/training/modal/openbrowser_modal.py::eval_mdpo_fsdfm_mu8

Cost (preemptible, ~3 hr/job)
-----------------------------
    Training: 3x L40S + 3x A10 ~$34
    Eval: 3x L40S + 3x A10 ~$20 (shorter per-job)
"""

import logging
import os
import subprocess

import modal
from modal import App, Image as ModalImage, Volume, Secret, FilePatternMatcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# GPU mapping (Anyscale -> Modal)
#   g6e.xlarge (L40S 48GB) -> "L40S"
#   g5.xlarge  (A10G 24GB) -> "A10"
GPU_REFUSION = "L40S"
GPU_FSDFM = "A10"

# Volume mount -- matches Anyscale /mnt/user_storage so persist_checkpoint()
# and all checkpoint paths work without code changes.
VOLUME_MOUNT = "/mnt/user_storage"

# Timeouts (seconds)
TRAIN_TIMEOUT = 14400  # 4 hours
EVAL_TIMEOUT = 7200  # 2 hours
DOWNLOAD_TIMEOUT = 3600  # 1 hour

# HuggingFace repos for SFT checkpoints
HF_REFUSION_SFT = "billyenrizky/ReFusion-8B-SFT"
HF_FSDFM_SFT = "billyenrizky/FS-DFM-1.3B-SFT"

# Checkpoint paths on volume
# Download paths -- where snapshot_download saves the HF repo
REFUSION_SFT_DOWNLOAD = f"{VOLUME_MOUNT}/openbrowser/checkpoints/flow-llm-sft/adapter"
FSDFM_SFT_DOWNLOAD = f"{VOLUME_MOUNT}/openbrowser/checkpoints/fsdfm-sft"
# Trainer paths -- where the actual model files live after download
# (HF repos have nested subdirectories: adapter/adapter/ and lora_adapter/)
REFUSION_SFT_PATH = f"{REFUSION_SFT_DOWNLOAD}/adapter"
FSDFM_SFT_PATH = f"{FSDFM_SFT_DOWNLOAD}/lora_adapter"

# ===========================================================================
# MODAL PRIMITIVES
# ===========================================================================

app = App("openbrowser-training")
volume = Volume.from_name("openbrowser-checkpoints", create_if_missing=True)
secret = Secret.from_name("openbrowser-secrets")

# Container image -- replicates Containerfile.online-grpo
image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    # System deps for Chromium (from Containerfile.online-grpo)
    .apt_install(
        "libnss3", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2",
        "libxkbcommon0", "libxcomposite1", "libxdamage1", "libxrandr2", "libgbm1",
        "libasound2t64", "libpango-1.0-0", "libpangocairo-1.0-0", "libgtk-3-0",
        "fonts-liberation", "xdg-utils", "wget", "git", "build-essential", "curl",
    )
    # Python deps (matching Containerfile.online-grpo)
    .pip_install(
        "numpy>=2.0,<2.5",
        "scipy>=1.14",
        "pyarrow>=17.0",
        "torch>=2.0",
        "peft>=0.16",
        "datasets>=2.18",
        "accelerate>=1.0",
        "bitsandbytes>=0.43",
        "boto3",
        "flask",
        "playwright",
        "openbrowser-ai",
        "cdp-use",
        "bubus>=1.5.6",
        "httpx>=0.28.1",
        "websockets>=15.0.1",
        "langgraph",
        "langchain-core>=0.3.0",
        "pydantic>=2.0.0",
        "aiofiles",
        "transformers==4.52.4",
        "urllib3<2",
        "uv",
        "huggingface_hub",
    )
    # Copy project code into container
    .add_local_dir(
        local_path=".",
        remote_path="/root/openbrowser",
        copy=True,
        ignore=FilePatternMatcher(
            ".venv/",
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".env",
            ".cursor/",
            "presentation/",
            "node_modules/",
            "frontend/",
            "data/mind2web/",
            "data/webarena/",
            "data/formfactory/.git/",
            "data/formfactory/img/",
            "results/",
            "outputs/",
            "openbrowser_agent_data/",
            "local_docs/",
            "extension.zip",
            "STAD80/",
            "STAD68/",
            "CSC490/",
            "landing/",
        ),
    )
    .workdir("/root/openbrowser")
    # Install Chromium for Playwright and symlink
    .run_commands(
        "playwright install chromium",
        "CHROME_PATH=$(python -c \"from playwright.sync_api import sync_playwright; "
        "pw=sync_playwright().start(); print(pw.chromium.executable_path); pw.stop()\" "
        "2>/dev/null) && if [ -n \"$CHROME_PATH\" ] && [ -f \"$CHROME_PATH\" ]; "
        "then ln -sf \"$CHROME_PATH\" /usr/bin/chromium; fi",
    )
)

# ===========================================================================
# HELPERS
# ===========================================================================


def _run(cmd: str):
    """Shell out to bash, raise on failure."""
    logger.info(">>> %s", cmd)
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {cmd}")


def _download_formfactory():
    """Download FormFactory dataset (required for online RL evaluation)."""
    _run("python -m infra.eval.scripts.download_datasets --datasets formfactory")


def _run_trainer(module: str):
    """Download FormFactory then run a trainer module."""
    _download_formfactory()
    _run(f"python -m {module}")


# ===========================================================================
# ENVIRONMENT VARIABLE CONFIGS (from Anyscale YAMLs)
# ===========================================================================

_COMMON_REFUSION = {
    "FLOW_TRAIN_FILE": "data/processed/formfactory_sft.jsonl",
    "MAX_TRAIN_SAMPLES": "50",
    "NUM_EPOCHS": "1",
    "FLOW_LLM_SFT_CHECKPOINT": REFUSION_SFT_PATH,
    "GEN_STEPS": "64",
    "GEN_TEMP": "1.0",
    "SHUFFLE": "true",
    "MIN_NONZERO": "1",
    "EARLY_STOP_MAX_STEPS": "40",
    "GRAD_CLIP": "1.0",
}

_COMMON_FSDFM = {
    "FLOW_TRAIN_FILE": "data/processed/formfactory_sft.jsonl",
    "MAX_TRAIN_SAMPLES": "50",
    "NUM_EPOCHS": "1",
    "FSDFM_SFT_CHECKPOINT": FSDFM_SFT_PATH,
    "GEN_STEPS": "64",
    "GEN_TEMP": "1.0",
    "SHUFFLE": "true",
    "MIN_NONZERO": "1",
    "EARLY_STOP_MAX_STEPS": "40",
    "GRAD_CLIP": "1.0",
}

ENV_ESPO_REFUSION_MU8 = {
    **_COMMON_REFUSION,
    "GROUP_SIZE": "4",
    "LR": "1e-5",
    "KL_COEFF": "3e-3",
    "EPSILON_LOW": "0.2",
    "EPSILON_HIGH": "0.2",
    "MU": "8",
    "NUM_MC": "2",
    "COUPLED": "true",
}

ENV_ESPO_FSDFM_MU8 = {
    **_COMMON_FSDFM,
    "GROUP_SIZE": "4",
    "LR": "1e-5",
    "KL_COEFF": "3e-3",
    "EPSILON_LOW": "0.2",
    "EPSILON_HIGH": "0.2",
    "MU": "8",
    "NUM_MC": "2",
    "COUPLED": "true",
}

ENV_CJGRPO_REFUSION = {
    **_COMMON_REFUSION,
    "GROUP_SIZE": "4",
    "LR": "5e-5",
    "KL_COEFF": "0.04",
    "EPSILON": "0.2",
    "MU": "1",
    "NUM_SAMPLED_TIMESTEPS": "8",
    "CONFIDENCE_NOISE_STD": "0.1",
    "CHECKPOINT_EVERY_STEPS": "10",
}

ENV_CJGRPO_FSDFM = {
    **_COMMON_FSDFM,
    "GROUP_SIZE": "4",
    "LR": "1e-5",
    "KL_COEFF": "0.04",
    "EPSILON": "0.2",
    "MU": "1",
    "NUM_SAMPLED_TIMESTEPS": "8",
    "CHECKPOINT_EVERY_STEPS": "5",
}

ENV_MDPO_REFUSION = {
    **_COMMON_REFUSION,
    "GROUP_SIZE": "4",
    "LR": "1e-5",
    "KL_COEFF": "0.04",
    "EPSILON": "0.2",
    "MU": "1",
    "SAMPLE_TRAIN_STEPS": "8",
    "WARMUP_STEPS": "5",
    "CONFIDENCE_NOISE_STD": "0.1",
    "CHECKPOINT_EVERY_STEPS": "10",
}

ENV_MDPO_FSDFM = {
    **_COMMON_FSDFM,
    "GROUP_SIZE": "4",
    "LR": "1e-5",
    "KL_COEFF": "0.04",
    "EPSILON": "0.2",
    "MU": "1",
    "SAMPLE_TRAIN_STEPS": "8",
    "WARMUP_STEPS": "5",
    "CHECKPOINT_EVERY_STEPS": "5",
}

# --- mu=8 variants for FS-DFM (stabilization experiment) ---
# CJ-GRPO and MDPO collapsed FS-DFM with mu=1. Root causes identified:
# 1. fp16 forward pass overflows (max 65504) through 21 transformer blocks,
#    producing NaN logits in ~83% of per-step log-prob computations.
#    bf16 (max 3.4e38) avoids this. The old "bf16 causes NaN in log-prob
#    chains" was pre-log_softmax refactor; F.log_softmax is bf16-safe.
# 2. All-zero rewards when stochastic generation fails -> no gradient signal
# 3. Per-step importance ratios amplify noise vs sequence-level (ESPO)
# Fixes: bf16 (default), greedy baseline rollout, StableDRL clipping retained.

ENV_CJGRPO_FSDFM_MU8 = {
    **ENV_CJGRPO_FSDFM,
    "MU": "8",
    "LR": "1e-6",           # 10x lower (StableDRL recommendation)
    "GRAD_CLIP": "0.2",     # 5x tighter (StableDRL recommendation)
}

ENV_MDPO_FSDFM_MU8 = {
    **ENV_MDPO_FSDFM,
    "MU": "8",
    "LR": "1e-6",           # 10x lower (StableDRL recommendation)
    "GRAD_CLIP": "0.2",     # 5x tighter (StableDRL recommendation)
}


# ===========================================================================
# CHECKPOINT DOWNLOAD (one-time setup)
# ===========================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT,
)
def download_sft_checkpoints():
    """Download SFT checkpoints from HuggingFace to the Modal Volume.

    Run once before training:
        modal run infra/training/modal/openbrowser_modal.py::download_sft_checkpoints
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")

    # ReFusion SFT
    if os.path.exists(REFUSION_SFT_DOWNLOAD) and os.listdir(REFUSION_SFT_DOWNLOAD):
        logger.info("ReFusion SFT already exists at %s, skipping.", REFUSION_SFT_DOWNLOAD)
    else:
        logger.info("Downloading %s -> %s", HF_REFUSION_SFT, REFUSION_SFT_DOWNLOAD)
        os.makedirs(REFUSION_SFT_DOWNLOAD, exist_ok=True)
        snapshot_download(
            repo_id=HF_REFUSION_SFT,
            local_dir=REFUSION_SFT_DOWNLOAD,
            token=token,
        )
        logger.info("ReFusion SFT download complete.")

    # FS-DFM SFT
    if os.path.exists(FSDFM_SFT_DOWNLOAD) and os.listdir(FSDFM_SFT_DOWNLOAD):
        logger.info("FS-DFM SFT already exists at %s, skipping.", FSDFM_SFT_DOWNLOAD)
    else:
        logger.info("Downloading %s -> %s", HF_FSDFM_SFT, FSDFM_SFT_DOWNLOAD)
        os.makedirs(FSDFM_SFT_DOWNLOAD, exist_ok=True)
        snapshot_download(
            repo_id=HF_FSDFM_SFT,
            local_dir=FSDFM_SFT_DOWNLOAD,
            token=token,
        )
        logger.info("FS-DFM SFT download complete.")

    volume.commit()
    logger.info("SFT checkpoints ready on volume.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=1800,
    env=_COMMON_FSDFM,
)
def diagnose_fsdfm():
    """Diagnostic: test FS-DFM model loading and generation on Modal.

    Isolates whether the '!!!!' generation bug is caused by:
    (a) Base model loading failure
    (b) LoRA injection breaking the model
    (c) SFT LoRA weight loading failure
    (d) Generation function bug

    Run: modal run infra/training/modal/openbrowser_modal.py::diagnose_fsdfm
    """
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer

    from infra.training.flow_matching.config import FSDFM_MODEL_CONFIG
    from infra.training.flow_matching.fsdfm_model import (
        PolynomialConvexScheduler,
        generate_with_prefix_conditioning,
        inject_lora,
        load_fsdfm_from_huggingface,
        load_lora_weights,
    )

    model_config = FSDFM_MODEL_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DIAG: device=%s, CUDA available=%s", device, torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check what token ID '!' maps to
    excl_id = tokenizer.encode("!", add_special_tokens=False)
    logger.info("DIAG: '!' encodes to token IDs: %s", excl_id)
    logger.info("DIAG: token 0 decodes to: %r", tokenizer.decode([0]))

    scheduler = PolynomialConvexScheduler(
        exponent=model_config.get("scheduler_exponent", 2.0)
    )

    # --- Step 1: Load base model (no LoRA) ---
    logger.info("DIAG STEP 1: Loading base model (no LoRA)...")
    base_model = load_fsdfm_from_huggingface(model_config, device=device, dtype=torch.float16)

    test_prompt = "Fill in the form fields for a job application. Name: John Smith, Email: john@example.com"
    enc = tokenizer(
        test_prompt, add_special_tokens=True, truncation=True,
        max_length=512, return_tensors="pt",
    )
    prefix_ids = enc["input_ids"].to(device)
    gen_len = 256

    base_model.eval()
    with torch.no_grad():
        # Quick logit check: are logits degenerate?
        noise = torch.randint(0, model_config["vocab_size"], (1, gen_len), device=device)
        x_test = torch.cat([prefix_ids, noise], dim=1)
        t_test = torch.full((1,), 0.5, device=device, dtype=torch.float32)
        logits = base_model(x_test, t_test)
        logger.info("DIAG: Base model logits shape: %s, dtype: %s", logits.shape, logits.dtype)
        logger.info("DIAG: Base model logits stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                     logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
        # Check argmax distribution
        argmax_tokens = logits[0, prefix_ids.shape[1]:, :].argmax(dim=-1)
        unique_tokens = argmax_tokens.unique()
        logger.info("DIAG: Base model argmax unique tokens: %d / %d positions", len(unique_tokens), gen_len)
        logger.info("DIAG: Base model argmax first 20: %s", argmax_tokens[:20].tolist())
        del logits, noise, x_test

    gen_ids = generate_with_prefix_conditioning(
        model=base_model, prefix_ids=prefix_ids, gen_length=gen_len,
        config={**model_config, "num_sampling_steps": 64},
        scheduler=scheduler, temperature=0.0,
    )
    base_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    logger.info("DIAG STEP 1 RESULT (base, greedy, 300 chars): %.300s", base_text)

    gen_ids_stoch = generate_with_prefix_conditioning(
        model=base_model, prefix_ids=prefix_ids, gen_length=gen_len,
        config={**model_config, "num_sampling_steps": 64},
        scheduler=scheduler, temperature=1.0,
    )
    base_text_stoch = tokenizer.decode(gen_ids_stoch[0], skip_special_tokens=True)
    logger.info("DIAG STEP 1 RESULT (base, stochastic, 300 chars): %.300s", base_text_stoch)

    del base_model
    torch.cuda.empty_cache()

    # --- Step 2: Load model + inject LoRA (no SFT weights) ---
    logger.info("DIAG STEP 2: Loading model with fresh LoRA (no SFT)...")
    model_lora = load_fsdfm_from_huggingface(model_config, device=device, dtype=torch.float16)
    model_lora = inject_lora(model_lora, model_config)

    model_lora.eval()
    gen_ids = generate_with_prefix_conditioning(
        model=model_lora, prefix_ids=prefix_ids, gen_length=gen_len,
        config={**model_config, "num_sampling_steps": 64},
        scheduler=scheduler, temperature=0.0,
    )
    lora_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    logger.info("DIAG STEP 2 RESULT (fresh LoRA, greedy, 300 chars): %.300s", lora_text)

    # --- Step 3: Load SFT LoRA weights ---
    sft_checkpoint = os.environ.get("FSDFM_SFT_CHECKPOINT", "")
    logger.info("DIAG STEP 3: SFT checkpoint path: %s", sft_checkpoint)
    logger.info("DIAG STEP 3: Path exists: %s", Path(sft_checkpoint).exists() if sft_checkpoint else "N/A")

    if sft_checkpoint and Path(sft_checkpoint).exists():
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        logger.info("DIAG STEP 3: lora_weights.pt exists: %s", lora_path.exists())
        if lora_path.exists():
            # Check file size
            file_size = lora_path.stat().st_size
            logger.info("DIAG STEP 3: lora_weights.pt size: %d bytes (%.2f MB)", file_size, file_size / 1e6)

            # Load and inspect
            lora_state = torch.load(str(lora_path), map_location="cpu", weights_only=True)
            logger.info("DIAG STEP 3: Saved LoRA keys (%d): %s", len(lora_state), list(lora_state.keys())[:10])

            # Check model state dict keys
            model_keys = [k for k in model_lora.state_dict().keys() if "lora" in k]
            logger.info("DIAG STEP 3: Model LoRA keys (%d): %s", len(model_keys), model_keys[:10])

            # Check key overlap
            saved_keys = set(lora_state.keys())
            model_key_set = set(model_lora.state_dict().keys())
            matching = saved_keys & model_key_set
            missing_in_model = saved_keys - model_key_set
            logger.info("DIAG STEP 3: Matching keys: %d / %d saved", len(matching), len(saved_keys))
            if missing_in_model:
                logger.error("DIAG STEP 3: Keys in checkpoint but NOT in model: %s", list(missing_in_model)[:10])

            load_lora_weights(model_lora, str(lora_path))

            model_lora.eval()
            gen_ids = generate_with_prefix_conditioning(
                model=model_lora, prefix_ids=prefix_ids, gen_length=gen_len,
                config={**model_config, "num_sampling_steps": 64},
                scheduler=scheduler, temperature=0.0,
            )
            sft_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            logger.info("DIAG STEP 3 RESULT (SFT LoRA, greedy, 300 chars): %.300s", sft_text)

            gen_ids_stoch = generate_with_prefix_conditioning(
                model=model_lora, prefix_ids=prefix_ids, gen_length=gen_len,
                config={**model_config, "num_sampling_steps": 64},
                scheduler=scheduler, temperature=1.0,
            )
            sft_text_stoch = tokenizer.decode(gen_ids_stoch[0], skip_special_tokens=True)
            logger.info("DIAG STEP 3 RESULT (SFT LoRA, stochastic, 300 chars): %.300s", sft_text_stoch)
        else:
            logger.error("DIAG STEP 3: lora_weights.pt NOT FOUND at %s", lora_path)
    else:
        logger.error("DIAG STEP 3: SFT checkpoint dir not found: %s", sft_checkpoint)
        # List what's actually in the checkpoints directory
        ckpt_base = Path(VOLUME_MOUNT) / "openbrowser" / "checkpoints"
        if ckpt_base.exists():
            for p in sorted(ckpt_base.rglob("*")):
                if p.is_file():
                    logger.info("DIAG: Volume file: %s (%d bytes)", p, p.stat().st_size)

    logger.info("DIAG: Diagnostic complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=3600,
    env={
        **ENV_CJGRPO_FSDFM_MU8,
        "EARLY_STOP_MAX_STEPS": "2",
        "MAX_TRAIN_SAMPLES": "5",
    },
)
def test_cjgrpo_fsdfm():
    """Short CJ-GRPO test (2 steps) to see full logs from startup to collapse."""
    _download_formfactory()
    _run("python -m infra.training.flow_matching.cjgrpo_fsdfm_trainer")
    logger.info("test_cjgrpo_fsdfm complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=1800,
)
def diagnose_nan():
    """Diagnose autograd-specific NaN in FS-DFM 1.3B forward pass.

    Runs 11 controlled tests to identify:
    1. Whether NaN requires autograd graph (not just enable_grad context)
    2. Which transformer block first produces NaN under autograd
    3. Which sub-operation within that block triggers NaN
    4. Whether the block NaNs in isolation or only with accumulated graph
    5. Whether gradient checkpointing resolves the issue
    6. What torch.autograd.detect_anomaly reports

    Prerequisites:
        modal run infra/training/modal/openbrowser_modal.py::download_sft_checkpoints

    Run:
        modal run infra/training/modal/openbrowser_modal.py::diagnose_nan
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    from pathlib import Path
    from transformers import AutoTokenizer

    from infra.training.flow_matching.config import FSDFM_MODEL_CONFIG
    from infra.training.flow_matching.fsdfm_model import (
        PolynomialConvexScheduler,
        compute_discrete_step_log_prob,
        generate_with_prefix_conditioning_trajectory,
        inject_lora,
        load_fsdfm_from_huggingface,
        load_lora_weights,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = FSDFM_MODEL_CONFIG
    SEP = "=" * 70

    logger.info(SEP)
    logger.info("FS-DFM 1.3B AUTOGRAD NaN DIAGNOSTIC")
    logger.info(SEP)

    # --- Load model with SFT LoRA ---
    logger.info("Loading FS-DFM 1.3B (%s)...", dtype)
    model = load_fsdfm_from_huggingface(config, device=device, dtype=dtype)
    model = inject_lora(model, config)

    sft_path = Path(FSDFM_SFT_PATH) / "lora_weights.pt"
    if not sft_path.exists():
        logger.error("SFT weights NOT FOUND at %s", sft_path)
        ckpt_base = Path(VOLUME_MOUNT) / "openbrowser" / "checkpoints"
        if ckpt_base.exists():
            for p in sorted(ckpt_base.rglob("*"))[:30]:
                if p.is_file():
                    logger.info("  Volume: %s (%d bytes)", p, p.stat().st_size)
        logger.error("Run download_sft_checkpoints first.")
        return

    load_lora_weights(model, str(sft_path))
    model.eval()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded. Trainable: %d (%.2fM)", n_train, n_train / 1e6)

    # --- Deterministic test input ---
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Fill in the form fields for a job application."
    prefix = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    noise = torch.randint(0, config["vocab_size"], (1, 256), device=device)
    x_test = torch.cat([prefix, noise], dim=1)
    t_test = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    logger.info("Test input: shape=%s, t=0.5", list(x_test.shape))

    # --- Helper ---
    def check(logits, label):
        has_nan = torch.isnan(logits).any().item()
        if has_nan:
            pct = torch.isnan(logits).float().mean().item() * 100
            v = logits[~torch.isnan(logits)]
            rng = "[%.4f, %.4f]" % (v.min().item(), v.max().item()) if v.numel() else "all-NaN"
            logger.info("  %s: NaN=YES (%.1f%%), valid range=%s", label, pct, rng)
        else:
            logger.info("  %s: NaN=NO, range=[%.4f, %.4f], std=%.4f",
                        label, logits.min().item(), logits.max().item(), logits.std().item())
        return has_nan

    # ==============================================================
    # TEST 1: no_grad (control -- must be NaN-free)
    # ==============================================================
    logger.info("\n--- TEST 1: torch.no_grad() [control] ---")
    with torch.no_grad():
        check(model(x_test, t_test), "no_grad")
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 2: enable_grad + ALL params frozen (no graph built)
    # ==============================================================
    logger.info("\n--- TEST 2: enable_grad, all params frozen (no graph) ---")
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in lora_params:
        p.requires_grad = False
    test2_nan = check(model(x_test, t_test), "grad_ctx_no_graph")
    for _, p in lora_params:
        p.requires_grad = True
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 3: enable_grad + LoRA grad (reproduce the bug)
    # ==============================================================
    logger.info("\n--- TEST 3: enable_grad, LoRA requires_grad=True ---")
    test3_nan = check(model(x_test, t_test), "grad_with_lora")
    torch.cuda.empty_cache()

    if not test3_nan:
        if test2_nan:
            logger.info("NaN in TEST 2 (no graph) but not TEST 3 (with graph) -- unexpected.")
        else:
            logger.info("NaN not reproduced with random noise. Testing trajectory states...")

        # ==============================================================
        # TEST 3b: Generate trajectory under no_grad (Euler solver)
        # ==============================================================
        logger.info("\n--- TEST 3b: Generate trajectory via Euler solver ---")
        scheduler = PolynomialConvexScheduler(exponent=2.0)
        traj_config = {
            **config,
            "num_generation_steps": 10,  # Fewer steps for diagnostic speed
        }
        with torch.no_grad():
            trajectory = generate_with_prefix_conditioning_trajectory(
                model=model,
                prefix_ids=prefix,
                gen_length=128,
                config=traj_config,
                scheduler=scheduler,
                temperature=1.0,
            )
        n_steps = len(trajectory.steps)
        logger.info("Trajectory: %d steps, final_tokens=%s, edit_mask sum=%d",
                     n_steps, list(trajectory.final_tokens.shape),
                     trajectory.edit_mask.sum().item())
        for i, step in enumerate(trajectory.steps):
            n_same = (step.x_t == step.x_next).sum().item()
            logger.info("  step %d: t=%.4f, x_t=%s, same_tokens=%d/%d",
                         i, step.t_value, list(step.x_t.shape),
                         n_same, step.x_t.shape[1])
        torch.cuda.empty_cache()

        # ==============================================================
        # TEST 3c: Forward pass on each trajectory step under enable_grad
        # ==============================================================
        logger.info("\n--- TEST 3c: Forward pass on trajectory states (enable_grad) ---")
        traj_nan_step = None
        for i, step in enumerate(trajectory.steps):
            t_i = torch.full((1,), step.t_value, device=device, dtype=torch.float32)
            logits_i = model(step.x_t, t_i)
            has_nan = torch.isnan(logits_i).any().item()
            if has_nan:
                pct = torch.isnan(logits_i).float().mean().item() * 100
                v = logits_i[~torch.isnan(logits_i)]
                rng = "[%.4f, %.4f]" % (v.min().item(), v.max().item()) if v.numel() else "all-NaN"
                logger.info("  step %d (t=%.4f): NaN=YES (%.1f%%), valid range=%s",
                             i, step.t_value, pct, rng)
                if traj_nan_step is None:
                    traj_nan_step = i
            else:
                logger.info("  step %d (t=%.4f): NaN=NO, range=[%.4f, %.4f], std=%.4f",
                             i, step.t_value,
                             logits_i.min().item(), logits_i.max().item(),
                             logits_i.std().item())
            del logits_i
            torch.cuda.empty_cache()

        # ==============================================================
        # TEST 3d: compute_discrete_step_log_prob on each trajectory step
        # ==============================================================
        logger.info("\n--- TEST 3d: compute_discrete_step_log_prob on trajectory ---")
        dt = 1.0 / traj_config["num_generation_steps"]
        response_mask = trajectory.edit_mask.float()
        logprob_nan_step = None
        for i, step in enumerate(trajectory.steps):
            try:
                lp = compute_discrete_step_log_prob(
                    model=model,
                    x_t=step.x_t,
                    x_next=step.x_next,
                    t_scalar=step.t_value,
                    dt=dt,
                    scheduler=scheduler,
                    vocab_size=config["vocab_size"],
                    response_mask=response_mask,
                    temperature=1.0,
                )
                has_nan = torch.isnan(lp).any().item()
                if has_nan:
                    logger.info("  step %d (t=%.4f): log_prob NaN=YES, val=%s",
                                 i, step.t_value, lp.tolist())
                    if logprob_nan_step is None:
                        logprob_nan_step = i
                else:
                    logger.info("  step %d (t=%.4f): log_prob=%.6f",
                                 i, step.t_value, lp.item())
            except Exception as e:
                logger.error("  step %d (t=%.4f): EXCEPTION: %s",
                              i, step.t_value, str(e)[:300])
                if logprob_nan_step is None:
                    logprob_nan_step = i
            model.zero_grad()
            torch.cuda.empty_cache()

        # Determine which input to use for remaining tests (4-11)
        if traj_nan_step is not None:
            logger.info("\nUsing trajectory step %d for block-level tests (forward NaN).",
                         traj_nan_step)
            nan_step = trajectory.steps[traj_nan_step]
            x_test = nan_step.x_t
            t_test = torch.full((1,), nan_step.t_value, device=device, dtype=torch.float32)
        elif logprob_nan_step is not None:
            logger.info("\nForward clean but log_prob NaN at step %d. "
                         "NaN is in post-processing, not model forward.",
                         logprob_nan_step)
            # Still test with this step's input for completeness
            nan_step = trajectory.steps[logprob_nan_step]
            x_test = nan_step.x_t
            t_test = torch.full((1,), nan_step.t_value, device=device, dtype=torch.float32)
        else:
            logger.info("\nNo NaN found in trajectory forward or log_prob. "
                         "Testing full training loop simulation...")

        # ==============================================================
        # TEST 3e: Simulate CJ-GRPO training loop
        # Accumulate gradients across K timesteps, optimizer step, then
        # check if the next forward pass produces NaN (weight corruption).
        # ==============================================================
        logger.info("\n--- TEST 3e: Full training loop simulation ---")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

        # Cache old log-probs (frozen) for importance ratio
        logger.info("  Caching old log-probs under no_grad...")
        old_lps = []
        with torch.no_grad():
            for step in trajectory.steps:
                lp = compute_discrete_step_log_prob(
                    model=model, x_t=step.x_t, x_next=step.x_next,
                    t_scalar=step.t_value, dt=dt,
                    scheduler=scheduler, vocab_size=config["vocab_size"],
                    response_mask=response_mask, temperature=1.0,
                )
                old_lps.append(lp.detach())
        logger.info("  Cached %d old log-probs: %s",
                     len(old_lps), [f"{lp.item():.4f}" for lp in old_lps])

        # Simulate 10 optimizer steps to stress test the fix
        num_sim_steps = 10
        num_sampled = min(8, len(trajectory.steps))
        sim_nan_found = False

        for sim_step in range(num_sim_steps):
            optimizer.zero_grad()
            step_losses = []

            # Sample K timesteps
            import random
            random.seed(sim_step)
            sampled_idx = sorted(random.sample(range(len(trajectory.steps)), num_sampled))

            for idx in sampled_idx:
                step = trajectory.steps[idx]
                cur_lp = compute_discrete_step_log_prob(
                    model=model, x_t=step.x_t, x_next=step.x_next,
                    t_scalar=step.t_value, dt=dt,
                    scheduler=scheduler, vocab_size=config["vocab_size"],
                    response_mask=response_mask, temperature=1.0,
                )

                if torch.isnan(cur_lp).any() or torch.isinf(cur_lp).any():
                    logger.info("  sim_step %d, t=%.4f: cur_log_prob NaN/Inf!",
                                 sim_step, step.t_value)
                    sim_nan_found = True
                    break

                # Simplified CJ-GRPO loss: policy_loss + kl_loss
                log_ratio = cur_lp - old_lps[idx]
                ratio = torch.exp(log_ratio)
                # Non-uniform advantages (stress test -- real training has varied rewards)
                advantage = 1.0 - 2.0 * (idx / max(len(trajectory.steps) - 1, 1))
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -(clipped_ratio * advantage).mean()

                kl_diff = old_lps[idx] - cur_lp  # ref ~ old for sim
                kl_loss = (kl_diff ** 2 / 2.0).mean()

                loss = (policy_loss + 0.04 * kl_loss) / num_sampled
                loss.backward()
                step_losses.append(loss.item())

            if sim_nan_found:
                break

            # Check gradients before clipping
            grad_norms = {}
            any_grad_nan = False
            for name, p in model.named_parameters():
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    has_nan = torch.isnan(p.grad).any().item()
                    if has_nan:
                        any_grad_nan = True
                    # Log only LoRA params (trainable)
                    if p.requires_grad:
                        grad_norms[name] = (gn, has_nan)

            total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            logger.info("  sim_step %d: losses=%s, grad_norm=%.4f, grad_nan=%s",
                         sim_step, [f"{l:.6f}" for l in step_losses],
                         total_grad_norm.item(), any_grad_nan)

            if any_grad_nan:
                logger.info("  NaN in gradients! Param details:")
                for name, (gn, has_nan) in sorted(grad_norms.items()):
                    if has_nan:
                        logger.info("    %s: grad_norm=%.6f, NaN=True", name, gn)
                sim_nan_found = True
                break

            # Optimizer step
            optimizer.step()

            # Check weights after optimizer step
            any_weight_nan = False
            for name, p in model.named_parameters():
                if p.requires_grad and torch.isnan(p).any():
                    any_weight_nan = True
                    logger.info("  Weight NaN after sim_step %d: %s", sim_step, name)

            if any_weight_nan:
                logger.info("  Optimizer step corrupted weights!")
                sim_nan_found = True
                break

            # Forward pass after optimizer step to check for NaN
            for i, step in enumerate(trajectory.steps[:3]):
                t_i = torch.full((1,), step.t_value, device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits_ng = model(step.x_t, t_i)
                logits_g = model(step.x_t, t_i)

                ng_nan = torch.isnan(logits_ng).any().item()
                g_nan = torch.isnan(logits_g).any().item()
                logger.info("  post-step%d fwd step%d(t=%.2f): "
                             "no_grad_nan=%s, grad_nan=%s, "
                             "ng_range=[%.2f,%.2f], g_range=[%.2f,%.2f]",
                             sim_step, i, step.t_value,
                             ng_nan, g_nan,
                             logits_ng.min().item(), logits_ng.max().item(),
                             logits_g.min().item(), logits_g.max().item())
                if g_nan or ng_nan:
                    sim_nan_found = True
                del logits_ng, logits_g
            torch.cuda.empty_cache()

            if sim_nan_found:
                break

        if not sim_nan_found:
            logger.info("\n  Training loop simulation: NO NaN after %d optimizer steps.",
                         num_sim_steps)
            logger.info("  NaN may require real browser rewards (non-uniform advantages)"
                         " or more steps.")
            logger.info("DIAGNOSTIC COMPLETE -- NaN not reproduced.")
            return
        else:
            logger.info("\n  Training loop simulation: NaN REPRODUCED at sim_step %d!",
                         sim_step)

            # ==============================================================
            # TEST 3f: Retry with full float32 model (the fix)
            # bf16 backward through 21 blocks overflows on extreme gradients
            # from Poisson jump math. Convert entire model to float32.
            # ==============================================================
            logger.info("\n--- TEST 3f: Training loop with full float32 model ---")

            # Convert entire model to float32 and reload weights
            model = model.float()
            load_lora_weights(model, str(sft_path))
            model.eval()
            model.gradient_checkpointing = False  # test without checkpointing

            # Verify all params are float32
            param_dtypes = set(str(p.dtype) for p in model.parameters())
            logger.info("  Model param dtypes: %s", param_dtypes)

            # Refresh trainable params and re-cache old log-probs in float32
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            logger.info("  Re-caching old log-probs in float32...")
            old_lps = []
            with torch.no_grad():
                for step in trajectory.steps:
                    lp = compute_discrete_step_log_prob(
                        model=model, x_t=step.x_t, x_next=step.x_next,
                        t_scalar=step.t_value, dt=dt,
                        scheduler=scheduler, vocab_size=config["vocab_size"],
                        response_mask=response_mask, temperature=1.0,
                    )
                    old_lps.append(lp.detach())
            logger.info("  Old log-probs: %s", [f"{lp.item():.4f}" for lp in old_lps])

            gc_optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
            gc_nan_found = False

            for gc_step in range(num_sim_steps):
                gc_optimizer.zero_grad()
                gc_losses = []

                random.seed(gc_step)
                gc_sampled = sorted(random.sample(range(len(trajectory.steps)), num_sampled))

                for idx in gc_sampled:
                    step = trajectory.steps[idx]
                    cur_lp = compute_discrete_step_log_prob(
                        model=model, x_t=step.x_t, x_next=step.x_next,
                        t_scalar=step.t_value, dt=dt,
                        scheduler=scheduler, vocab_size=config["vocab_size"],
                        response_mask=response_mask, temperature=1.0,
                    )

                    if torch.isnan(cur_lp).any() or torch.isinf(cur_lp).any():
                        logger.info("  gc_step %d, t=%.4f: cur_log_prob NaN/Inf!",
                                     gc_step, step.t_value)
                        gc_nan_found = True
                        break

                    log_ratio = cur_lp - old_lps[idx]
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                    policy_loss = -(clipped_ratio * 1.0).mean()
                    kl_diff = old_lps[idx] - cur_lp
                    kl_loss = (kl_diff ** 2 / 2.0).mean()
                    loss = (policy_loss + 0.04 * kl_loss) / num_sampled
                    loss.backward()
                    gc_losses.append(loss.item())

                if gc_nan_found:
                    break

                gc_any_nan = any(
                    torch.isnan(p.grad).any().item()
                    for p in model.parameters() if p.grad is not None
                )
                gc_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                logger.info("  gc_step %d: losses=%s, grad_norm=%.4f, grad_nan=%s",
                             gc_step, [f"{l:.6f}" for l in gc_losses],
                             gc_grad_norm.item(), gc_any_nan)

                if gc_any_nan:
                    gc_nan_found = True
                    break

                gc_optimizer.step()

                # Verify forward still clean
                t_chk = torch.full((1,), 0.5, device=device, dtype=torch.float32)
                logits_chk = model(trajectory.steps[0].x_t, t_chk)
                fwd_nan = torch.isnan(logits_chk).any().item()
                logger.info("  gc_step %d post-optim fwd: nan=%s, range=[%.2f,%.2f]",
                             gc_step, fwd_nan,
                             logits_chk.min().item(), logits_chk.max().item())
                del logits_chk
                torch.cuda.empty_cache()

                if fwd_nan:
                    gc_nan_found = True
                    break

            if gc_nan_found:
                logger.info("  Gradient checkpointing did NOT fix NaN.")
            else:
                logger.info("  Gradient checkpointing FIXES NaN! "
                             "%d optimizer steps completed cleanly.", num_sim_steps)

            # Disable for remaining tests
            model.gradient_checkpointing = False

        # ==============================================================
        # TEST 3g: Pinpoint NaN in backward -- single step, detect_anomaly,
        # retain_grad on every intermediate in the Poisson math chain.
        # ==============================================================
        logger.info("\n--- TEST 3g: Single-step backward with detect_anomaly ---")

        # Reload fresh model to avoid corrupted weights
        model_3g = load_fsdfm_from_huggingface(config, device=device, dtype=dtype)
        model_3g = inject_lora(model_3g, config)
        load_lora_weights(model_3g, str(sft_path))
        model_3g.eval()

        # Test each trajectory step individually
        first_nan_intermediate = None
        param_nan = False
        for step_idx, step in enumerate(trajectory.steps):
            model_3g.zero_grad()

            t_val = torch.full((1,), step.t_value, device=device, dtype=torch.float32)

            # Forward pass in float32
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                logits = model_3g(step.x_t, t_val)  # [1, L, V]

            # Inline Poisson math with retain_grad on intermediates
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs.retain_grad()

            log_p_current = log_probs.gather(2, step.x_t.unsqueeze(-1)).squeeze(-1)
            log_p_current.retain_grad()

            log_p_jumped_to = log_probs.gather(2, step.x_next.unsqueeze(-1)).squeeze(-1)
            log_p_jumped_to.retain_grad()

            p_current = log_p_current.exp()
            p_current.retain_grad()

            sched = scheduler(t_val)
            alpha_t = sched["alpha_t"]
            d_alpha_t = sched["d_alpha_t"]
            rate_scale = d_alpha_t / (1.0 - alpha_t).clamp(min=1e-6)

            lambda_i = rate_scale.unsqueeze(-1) * (1.0 - p_current)
            lambda_i.retain_grad()

            stayed = (step.x_next == step.x_t)

            # Stay branch
            log_prob_stay = -lambda_i * dt
            log_prob_stay.retain_grad()

            # Jump branch
            neg_exp_term = torch.exp(-lambda_i * dt)
            neg_exp_term.retain_grad()

            log_jump_base = torch.log1p(-neg_exp_term.clamp(max=1.0 - 1e-4))
            log_jump_base.retain_grad()

            log_one_minus_p_current = torch.log1p(-p_current.clamp(max=1.0 - 1e-4))
            log_one_minus_p_current.retain_grad()

            log_prob_jump = log_jump_base + log_p_jumped_to - log_one_minus_p_current
            log_prob_jump.retain_grad()

            log_prob_per_pos = torch.where(stayed, log_prob_stay, log_prob_jump)
            log_prob_per_pos.retain_grad()

            log_prob_per_pos_masked = log_prob_per_pos * response_mask
            num_resp = response_mask.sum(dim=-1).clamp(min=1)
            log_prob = log_prob_per_pos_masked.sum(dim=-1) / num_resp

            # Check forward intermediate values
            n_stayed = stayed.sum().item()
            n_jumped = (~stayed).sum().item()
            n_resp = int(response_mask.sum().item())

            # Key diagnostics: check for extreme values that cause backward NaN
            p_current_vals = p_current[0, response_mask[0].bool()]
            p_near_1 = (p_current_vals > 1.0 - 1e-7).sum().item()
            p_exactly_1 = (p_current_vals == 1.0).sum().item()
            p_min = p_current_vals.min().item()
            p_max = p_current_vals.max().item()

            lambda_vals = lambda_i[0, response_mask[0].bool()]
            lambda_near_0 = (lambda_vals.abs() < 1e-6).sum().item()
            lambda_exactly_0 = (lambda_vals == 0.0).sum().item()

            neg_exp_vals = neg_exp_term[0, response_mask[0].bool()]
            neg_exp_near_1 = (neg_exp_vals > 1.0 - 1e-7).sum().item()
            neg_exp_exactly_1 = (neg_exp_vals == 1.0).sum().item()

            # For jumped positions: check 1 - neg_exp_term (denominator-ish)
            jumped_mask = (~stayed[0]) & response_mask[0].bool()
            if jumped_mask.any():
                one_minus_negexp_jumped = (1.0 - neg_exp_vals[jumped_mask[response_mask[0].bool()]])
                omp_min = one_minus_negexp_jumped.min().item()
                omp_near_0 = (one_minus_negexp_jumped.abs() < 1e-7).sum().item()
            else:
                omp_min = float("nan")
                omp_near_0 = 0

            logger.info("  step %d (t=%.4f): resp=%d, stayed=%d, jumped=%d, "
                         "rate_scale=%.4f",
                         step_idx, step.t_value, n_resp, n_stayed, n_jumped,
                         rate_scale.item())
            logger.info("    p_current: min=%.8f, max=%.8f, near_1(>1-1e-7)=%d, "
                         "exactly_1=%d",
                         p_min, p_max, p_near_1, p_exactly_1)
            logger.info("    lambda_i: near_0(<1e-6)=%d, exactly_0=%d",
                         lambda_near_0, lambda_exactly_0)
            logger.info("    neg_exp: near_1(>1-1e-7)=%d, exactly_1=%d",
                         neg_exp_near_1, neg_exp_exactly_1)
            logger.info("    jumped positions: 1-neg_exp min=%.8e, near_0=%d",
                         omp_min, omp_near_0)
            logger.info("    log_prob forward=%.6f, nan=%s",
                         log_prob.item(), torch.isnan(log_prob).any().item())

            # CJ-GRPO style loss for single step
            loss = -log_prob.mean()

            # Backward with detect_anomaly
            try:
                with torch.autograd.detect_anomaly(check_nan=True):
                    loss.backward()
                bwd_error = None
            except RuntimeError as e:
                bwd_error = str(e)[:500]

            # Check gradients on intermediates
            intermediates = {
                "log_probs": log_probs,
                "log_p_current": log_p_current,
                "log_p_jumped_to": log_p_jumped_to,
                "p_current": p_current,
                "lambda_i": lambda_i,
                "log_prob_stay": log_prob_stay,
                "neg_exp_term": neg_exp_term,
                "log_jump_base": log_jump_base,
                "log_one_minus_p_current": log_one_minus_p_current,
                "log_prob_jump": log_prob_jump,
                "log_prob_per_pos": log_prob_per_pos,
            }
            first_nan_intermediate = None
            for iname, itensor in intermediates.items():
                if itensor.grad is not None:
                    g = itensor.grad
                    has_nan = torch.isnan(g).any().item()
                    has_inf = torch.isinf(g).any().item()
                    g_clean = g[~torch.isnan(g) & ~torch.isinf(g)]
                    g_min = g_clean.min().item() if g_clean.numel() else float("nan")
                    g_max = g_clean.max().item() if g_clean.numel() else float("nan")
                    nan_count = torch.isnan(g).sum().item()
                    inf_count = torch.isinf(g).sum().item()

                    if has_nan and first_nan_intermediate is None:
                        first_nan_intermediate = iname
                        tag = " <<< FIRST NaN"
                    elif has_nan:
                        tag = " NaN"
                    elif has_inf:
                        tag = " INF"
                    else:
                        tag = ""

                    logger.info("    grad[%s]: nan=%d, inf=%d, "
                                 "range=[%.4e, %.4e]%s",
                                 iname, nan_count, inf_count, g_min, g_max, tag)

            # Check model param gradients
            param_nan = any(
                torch.isnan(p.grad).any().item()
                for p in model_3g.parameters() if p.grad is not None
            )

            if bwd_error:
                logger.info("    detect_anomaly error: %s", bwd_error[:300])
            logger.info("    first_nan_intermediate=%s, param_grad_nan=%s",
                         first_nan_intermediate, param_nan)

            # If this single step produces NaN, we found it
            if first_nan_intermediate or param_nan or bwd_error:
                logger.info("  >>> SINGLE STEP %d produces NaN in backward!", step_idx)

                # Extra: check which specific positions have NaN gradients
                if first_nan_intermediate and first_nan_intermediate in intermediates:
                    g = intermediates[first_nan_intermediate].grad
                    if g is not None and g.dim() >= 2:
                        resp_mask_bool = response_mask[0].bool()
                        # NaN positions
                        nan_pos = torch.isnan(g[0])
                        if nan_pos.dim() == 1:
                            nan_in_resp = (nan_pos & resp_mask_bool).sum().item()
                            nan_in_prefix = (nan_pos & ~resp_mask_bool).sum().item()
                            nan_in_stayed = (nan_pos & stayed[0]).sum().item()
                            nan_in_jumped = (nan_pos & ~stayed[0]).sum().item()
                            logger.info("    NaN positions: resp=%d, prefix=%d, "
                                         "stayed=%d, jumped=%d",
                                         nan_in_resp, nan_in_prefix,
                                         nan_in_stayed, nan_in_jumped)
                break  # Found the problematic step, stop
            else:
                logger.info("    CLEAN -- single step backward OK")

            del logits, log_probs, log_p_current, log_p_jumped_to, p_current
            del lambda_i, neg_exp_term, log_jump_base, log_one_minus_p_current
            del log_prob_stay, log_prob_jump, log_prob_per_pos, log_prob
            model_3g.zero_grad()
            torch.cuda.empty_cache()

        # If all individual steps clean, test accumulated backward
        if first_nan_intermediate is None and not param_nan:
            logger.info("\n  All individual steps clean. Testing accumulated backward...")
            model_3g.zero_grad()

            for step_idx, step in enumerate(trajectory.steps):
                lp = compute_discrete_step_log_prob(
                    model=model_3g, x_t=step.x_t, x_next=step.x_next,
                    t_scalar=step.t_value, dt=dt,
                    scheduler=scheduler, vocab_size=config["vocab_size"],
                    response_mask=response_mask, temperature=1.0,
                )
                loss = (-lp.mean()) / len(trajectory.steps)
                loss.backward()

                # Check gradients after each accumulated backward
                accum_nan = any(
                    torch.isnan(p.grad).any().item()
                    for p in model_3g.parameters() if p.grad is not None
                )
                if accum_nan:
                    logger.info("  >>> NaN appears after accumulating %d steps "
                                 "(step %d, t=%.4f)", step_idx + 1, step_idx,
                                 step.t_value)
                    break
            else:
                logger.info("  Accumulated backward across all steps: CLEAN")

        del model_3g
        torch.cuda.empty_cache()

        # Use post-corruption state for remaining tests
        t_test = torch.full((1,), 0.5, device=device, dtype=torch.float32)

    # ==============================================================
    # TEST 4: Per-block hooks -- which block first produces NaN?
    # ==============================================================
    logger.info("\n--- TEST 4: Per-block forward hooks ---")
    block_stats = {}
    hooks = []

    def make_hook(name):
        def fn(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            nan = torch.isnan(t).any().item()
            v = t[~torch.isnan(t)] if nan else t
            block_stats[name] = {
                "nan": nan,
                "min": v.min().item() if v.numel() else float("nan"),
                "max": v.max().item() if v.numel() else float("nan"),
                "std": v.std().item() if v.numel() > 1 else 0.0,
            }
        return fn

    hooks.append(model.vocab_embed.register_forward_hook(make_hook("embed")))
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(make_hook("block_%02d" % i)))
    hooks.append(model.output_layer.register_forward_hook(make_hook("output")))

    model(x_test, t_test)
    for h in hooks:
        h.remove()

    first_nan_name = None
    first_nan_idx = None
    order = ["embed"] + ["block_%02d" % i for i in range(len(model.blocks))] + ["output"]
    for name in order:
        s = block_stats[name]
        tag = ""
        if s["nan"] and first_nan_name is None:
            first_nan_name = name
            tag = " <<< FIRST NaN"
            if name.startswith("block_"):
                first_nan_idx = int(name.split("_")[1])
        elif s["nan"]:
            tag = " NaN"
        logger.info("  %-12s [%12.4f, %12.4f] std=%10.4f%s",
                     name, s["min"], s["max"], s["std"], tag)
    logger.info("First NaN: %s", first_nan_name)
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 5: Sub-layer hooks on first NaN block
    # ==============================================================
    if first_nan_idx is not None:
        logger.info("\n--- TEST 5: Sub-layer hooks on block_%02d ---", first_nan_idx)
        blk = model.blocks[first_nan_idx]
        sub_stats = {}
        sub_hooks = []

        def make_sub_hook(name):
            def fn(mod, inp, out):
                t = out if isinstance(out, torch.Tensor) else out[0]
                nan = torch.isnan(t).any().item()
                v = t[~torch.isnan(t)] if nan else t
                sub_stats[name] = {
                    "nan": nan,
                    "min": v.min().item() if v.numel() else float("nan"),
                    "max": v.max().item() if v.numel() else float("nan"),
                }
            return fn

        if first_nan_idx > 0:
            sub_hooks.append(model.blocks[first_nan_idx - 1].register_forward_hook(
                make_sub_hook("prev_block")))
        for sn in ["adaLN_modulation", "norm1", "qw", "kw", "vw",
                    "attn_out", "norm2", "mlp"]:
            if hasattr(blk, sn):
                sub_hooks.append(
                    getattr(blk, sn).register_forward_hook(make_sub_hook(sn)))
        sub_hooks.append(blk.register_forward_hook(make_sub_hook("block_out")))

        model(x_test, t_test)
        for h in sub_hooks:
            h.remove()

        sub_order = (["prev_block"] if first_nan_idx > 0 else []) + [
            "adaLN_modulation", "norm1", "qw", "kw", "vw",
            "attn_out", "norm2", "mlp", "block_out",
        ]
        for name in sub_order:
            if name in sub_stats:
                s = sub_stats[name]
                tag = " <<< NaN" if s["nan"] else ""
                logger.info("  %-20s [%12.4f, %12.4f]%s",
                            name, s["min"], s["max"], tag)
        torch.cuda.empty_cache()

    # ==============================================================
    # TEST 6: Isolated block -- does it NaN alone?
    # ==============================================================
    if first_nan_idx is not None:
        logger.info("\n--- TEST 6: Isolated block_%02d with enable_grad ---",
                    first_nan_idx)
        with torch.no_grad():
            x_inter = model.vocab_embed(x_test)
            c_inter = model.time_embedding(t_test)
            for i in range(first_nan_idx):
                x_inter = model.blocks[i](
                    x_inter, c_inter, model.rotary_cos, model.rotary_sin)
        # Detach: break graph from prior blocks
        x_inter = x_inter.detach()
        c_inter = c_inter.detach()
        # Run target block with autograd (LoRA params still require grad)
        out = model.blocks[first_nan_idx](
            x_inter, c_inter, model.rotary_cos, model.rotary_sin)
        iso_nan = torch.isnan(out).any().item()
        logger.info("  Isolated block_%02d NaN=%s", first_nan_idx, iso_nan)
        if iso_nan:
            logger.info("  >> NaN in single block alone (not accumulated)")
        else:
            logger.info("  >> NaN requires graph from prior blocks")
        del out, x_inter, c_inter
        torch.cuda.empty_cache()

    # ==============================================================
    # Gradient checkpointing helper
    # ==============================================================
    def _ckpt_forward(reentrant):
        x = model.vocab_embed(x_test)
        c = model.time_embedding(t_test)
        for blk in model.blocks:
            x = grad_checkpoint(
                blk, x, c, model.rotary_cos, model.rotary_sin,
                use_reentrant=reentrant)
        return model.output_layer(x, c)

    # ==============================================================
    # TEST 7: Gradient checkpointing (use_reentrant=True)
    # Forward runs under no_grad; backward re-runs with grad.
    # ==============================================================
    logger.info("\n--- TEST 7: Gradient ckpt (use_reentrant=True) ---")
    try:
        logits = _ckpt_forward(reentrant=True)
        t7_nan = check(logits, "ckpt_reentrant_True")
        if not t7_nan:
            logger.info("  >> reentrant=True FIXES forward NaN (forward ran under no_grad)")
            try:
                loss = logits.sum()
                loss.backward()
                bwd_nan = any(
                    torch.isnan(p.grad).any().item()
                    for p in model.parameters() if p.grad is not None)
                logger.info("  >> Backward gradient NaN=%s", bwd_nan)
            except RuntimeError as e:
                logger.error("  >> Backward FAILED: %s", str(e)[:300])
        model.zero_grad()
        del logits
    except Exception as e:
        logger.error("  TEST 7 error: %s", str(e)[:300])
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 8: Gradient checkpointing (use_reentrant=False)
    # Forward runs WITH grad; only intermediate storage changes.
    # ==============================================================
    logger.info("\n--- TEST 8: Gradient ckpt (use_reentrant=False) ---")
    try:
        logits = _ckpt_forward(reentrant=False)
        t8_nan = check(logits, "ckpt_reentrant_False")
        if not t8_nan:
            logger.info("  >> reentrant=False FIXES forward NaN")
            try:
                loss = logits.sum()
                loss.backward()
                bwd_nan = any(
                    torch.isnan(p.grad).any().item()
                    for p in model.parameters() if p.grad is not None)
                logger.info("  >> Backward gradient NaN=%s", bwd_nan)
            except RuntimeError as e:
                logger.error("  >> Backward FAILED: %s", str(e)[:300])
        model.zero_grad()
        del logits
    except Exception as e:
        logger.error("  TEST 8 error: %s", str(e)[:300])
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 9: torch.autograd.detect_anomaly
    # ==============================================================
    logger.info("\n--- TEST 9: detect_anomaly ---")
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            logits = model(x_test, t_test)
            loss = logits.nansum()
            loss.backward()
        logger.info("  detect_anomaly: completed without error")
    except RuntimeError as e:
        logger.info("  detect_anomaly error: %s", str(e)[:500])
    model.zero_grad()
    torch.cuda.empty_cache()

    # ==============================================================
    # TEST 10: adaLN modulation scale factors
    # ==============================================================
    logger.info("\n--- TEST 10: adaLN modulation scale factors ---")
    with torch.no_grad():
        c = model.time_embedding(t_test)
        for i, blk in enumerate(model.blocks):
            mod = blk.adaLN_modulation(F.silu(c))
            parts = mod.chunk(6, dim=-1)
            names = ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]
            vals = {n: (p.min().item(), p.max().item()) for n, p in zip(names, parts)}
            logger.info(
                "  block_%02d: scale1=[%.3f,%.3f] gate1=[%.3f,%.3f] "
                "scale2=[%.3f,%.3f] gate2=[%.3f,%.3f]",
                i,
                vals["scale1"][0], vals["scale1"][1],
                vals["gate1"][0], vals["gate1"][1],
                vals["scale2"][0], vals["scale2"][1],
                vals["gate2"][0], vals["gate2"][1],
            )

    # ==============================================================
    # TEST 11: LoRA weight ranges
    # ==============================================================
    logger.info("\n--- TEST 11: LoRA weight ranges ---")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("  %s: [%.6f, %.6f] std=%.6f",
                        name, p.min().item(), p.max().item(), p.std().item())

    logger.info("\n" + SEP)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info(SEP)


# ===========================================================================
# TRAINING JOBS
# ===========================================================================

# --- ESPO mu=8 ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=TRAIN_TIMEOUT,
    env=ENV_ESPO_REFUSION_MU8,
)
def espo_refusion_mu8():
    """ESPO mu=8 on ReFusion 8B (L40S)."""
    _run_trainer("infra.training.flow_matching.espo_refusion_trainer")
    volume.commit()
    logger.info("espo_refusion_mu8 complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=TRAIN_TIMEOUT,
    env=ENV_ESPO_FSDFM_MU8,
)
def espo_fsdfm_mu8():
    """ESPO mu=8 on FS-DFM 1.3B (A10)."""
    _run_trainer("infra.training.flow_matching.espo_fsdfm_trainer")
    volume.commit()
    logger.info("espo_fsdfm_mu8 complete.")


# --- CJ-GRPO ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=TRAIN_TIMEOUT,
    env=ENV_CJGRPO_REFUSION,
)
def cjgrpo_refusion():
    """CJ-GRPO on ReFusion 8B (L40S)."""
    _run_trainer("infra.training.flow_matching.cjgrpo_refusion_trainer")
    volume.commit()
    logger.info("cjgrpo_refusion complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=TRAIN_TIMEOUT,
    env=ENV_CJGRPO_FSDFM,
)
def cjgrpo_fsdfm():
    """CJ-GRPO on FS-DFM 1.3B (A10)."""
    _run_trainer("infra.training.flow_matching.cjgrpo_fsdfm_trainer")
    volume.commit()
    logger.info("cjgrpo_fsdfm complete.")


# --- MDPO ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=TRAIN_TIMEOUT,
    env=ENV_MDPO_REFUSION,
)
def mdpo_refusion():
    """MDPO on ReFusion 8B (L40S)."""
    _run_trainer("infra.training.flow_matching.mdpo_refusion_trainer")
    volume.commit()
    logger.info("mdpo_refusion complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=TRAIN_TIMEOUT,
    env=ENV_MDPO_FSDFM,
)
def mdpo_fsdfm():
    """MDPO on FS-DFM 1.3B (A10)."""
    _run_trainer("infra.training.flow_matching.mdpo_fsdfm_trainer")
    volume.commit()
    logger.info("mdpo_fsdfm complete.")


# --- mu=8 stabilization experiments (FS-DFM only) ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=TRAIN_TIMEOUT,
    env=ENV_CJGRPO_FSDFM_MU8,
)
def cjgrpo_fsdfm_mu8():
    """CJ-GRPO mu=8 on FS-DFM 1.3B (A10). Stabilization experiment."""
    _run_trainer("infra.training.flow_matching.cjgrpo_fsdfm_trainer")
    volume.commit()
    logger.info("cjgrpo_fsdfm_mu8 complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=TRAIN_TIMEOUT,
    env=ENV_MDPO_FSDFM_MU8,
)
def mdpo_fsdfm_mu8():
    """MDPO mu=8 on FS-DFM 1.3B (A10). Stabilization experiment."""
    _run_trainer("infra.training.flow_matching.mdpo_fsdfm_trainer")
    volume.commit()
    logger.info("mdpo_fsdfm_mu8 complete.")


# ===========================================================================
# EVALUATION JOBS
# ===========================================================================

# RL checkpoint paths on volume (from persist_checkpoint() calls in trainers)
CKPT_BASE = f"{VOLUME_MOUNT}/openbrowser/checkpoints"
ESPO_REFUSION_CKPT = f"{CKPT_BASE}/espo-refusion"
ESPO_FSDFM_CKPT = f"{CKPT_BASE}/espo-fsdfm"
CJGRPO_REFUSION_CKPT = f"{CKPT_BASE}/cjgrpo-refusion"
CJGRPO_FSDFM_CKPT = f"{CKPT_BASE}/cjgrpo-fsdfm"
MDPO_REFUSION_CKPT = f"{CKPT_BASE}/mdpo-refusion"
MDPO_FSDFM_CKPT = f"{CKPT_BASE}/mdpo-fsdfm-best-262-step27"


def _run_eval(eval_module: str, env_overrides: dict):
    """Run an eval script with env var overrides."""
    env_str = " ".join(f'{k}="{v}"' for k, v in env_overrides.items())
    _run(f"{env_str} python -m {eval_module}")


def _eval_refusion(checkpoint_path: str, result_prefix: str):
    """Run ReFusion eval on both val and test splits."""
    _download_formfactory()
    for split in ["val", "test"]:
        logger.info("Evaluating ReFusion: %s (%s)", result_prefix, split)
        _run_eval("infra.training.flow_matching.eval_refusion_sft", {
            "FLOW_LLM_SFT_CHECKPOINT": checkpoint_path,
            "EVAL_SPLIT": split,
            "MAX_EVAL_SAMPLES": "124",
            "EVAL_RESULT_NAME": f"{result_prefix}-{split}",
        })
        volume.commit()
        logger.info("Committed %s-%s results to volume", result_prefix, split)


def _eval_fsdfm(checkpoint_path: str, result_prefix: str):
    """Run FS-DFM eval on both val and test splits."""
    _download_formfactory()
    for split in ["val", "test"]:
        logger.info("Evaluating FS-DFM: %s (%s)", result_prefix, split)
        _run_eval("infra.training.flow_matching.eval_fsdfm_sft", {
            "FSDFM_SFT_CHECKPOINT": checkpoint_path,
            "EVAL_SPLIT": split,
            "MAX_EVAL_SAMPLES": "124",
            "EVAL_RESULT_NAME": f"{result_prefix}-{split}",
        })
        volume.commit()
        logger.info("Committed %s-%s results to volume", result_prefix, split)


# --- ESPO mu=8 evals ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=EVAL_TIMEOUT,
)
def eval_espo_refusion_mu8():
    """Eval ESPO mu=8 ReFusion on val+test (L40S)."""
    _eval_refusion(ESPO_REFUSION_CKPT, "espo-refusion-mu8")
    volume.commit()
    logger.info("eval_espo_refusion_mu8 complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=EVAL_TIMEOUT,
)
def eval_espo_fsdfm_mu8():
    """Eval ESPO mu=8 FS-DFM on val+test (A10)."""
    _eval_fsdfm(ESPO_FSDFM_CKPT, "espo-fsdfm-mu8")
    volume.commit()
    logger.info("eval_espo_fsdfm_mu8 complete.")


# --- CJ-GRPO evals ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=EVAL_TIMEOUT,
)
def eval_cjgrpo_refusion():
    """Eval CJ-GRPO ReFusion on val+test (L40S)."""
    _eval_refusion(CJGRPO_REFUSION_CKPT, "cjgrpo-refusion")
    volume.commit()
    logger.info("eval_cjgrpo_refusion complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=EVAL_TIMEOUT,
)
def eval_cjgrpo_fsdfm():
    """Eval CJ-GRPO FS-DFM on val+test (A10)."""
    _eval_fsdfm(CJGRPO_FSDFM_CKPT, "cjgrpo-fsdfm")
    volume.commit()
    logger.info("eval_cjgrpo_fsdfm complete.")


# --- MDPO evals ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_REFUSION,
    timeout=EVAL_TIMEOUT,
)
def eval_mdpo_refusion():
    """Eval MDPO ReFusion on val+test (L40S)."""
    _eval_refusion(MDPO_REFUSION_CKPT, "mdpo-refusion")
    volume.commit()
    logger.info("eval_mdpo_refusion complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=EVAL_TIMEOUT,
)
def eval_mdpo_fsdfm():
    """Eval MDPO FS-DFM on val+test (A10)."""
    _eval_fsdfm(MDPO_FSDFM_CKPT, "mdpo-fsdfm")
    volume.commit()
    logger.info("eval_mdpo_fsdfm complete.")


# --- mu=8 stabilization eval ---

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=EVAL_TIMEOUT,
)
def eval_cjgrpo_fsdfm_mu8():
    """Eval CJ-GRPO mu=8 FS-DFM on val+test (A10)."""
    _eval_fsdfm(CJGRPO_FSDFM_CKPT, "cjgrpo-fsdfm-mu8")
    volume.commit()
    logger.info("eval_cjgrpo_fsdfm_mu8 complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FSDFM,
    timeout=EVAL_TIMEOUT,
)
def eval_mdpo_fsdfm_mu8():
    """Eval MDPO mu=8 FS-DFM on val+test (A10)."""
    _eval_fsdfm(MDPO_FSDFM_CKPT, "mdpo-fsdfm-mu8")
    volume.commit()
    logger.info("eval_mdpo_fsdfm_mu8 complete.")


# ===========================================================================
# FULL PIPELINE (main entrypoint)
# ===========================================================================

@app.local_entrypoint()
def main():
    """Run all 6 training jobs.

    Usage: modal run infra/training/modal/openbrowser_modal.py
    """
    w = 70
    logger.info("=" * w)
    logger.info("OpenBrowser Training -- Modal Edition (6 jobs)")
    logger.info("  ESPO mu=8: ReFusion (L40S) + FS-DFM (A10)")
    logger.info("  CJ-GRPO:   ReFusion (L40S) + FS-DFM (A10)")
    logger.info("  MDPO:      ReFusion (L40S) + FS-DFM (A10)")
    logger.info("=" * w)

    # Submit all 6 jobs in parallel
    logger.info("[1/6] Submitting ESPO ReFusion mu=8 (L40S)...")
    h_espo_ref = espo_refusion_mu8.spawn()

    logger.info("[2/6] Submitting ESPO FS-DFM mu=8 (A10)...")
    h_espo_fsdfm = espo_fsdfm_mu8.spawn()

    logger.info("[3/6] Submitting CJ-GRPO ReFusion (L40S)...")
    h_cjgrpo_ref = cjgrpo_refusion.spawn()

    logger.info("[4/6] Submitting CJ-GRPO FS-DFM (A10)...")
    h_cjgrpo_fsdfm = cjgrpo_fsdfm.spawn()

    logger.info("[5/6] Submitting MDPO ReFusion (L40S)...")
    h_mdpo_ref = mdpo_refusion.spawn()

    logger.info("[6/6] Submitting MDPO FS-DFM (A10)...")
    h_mdpo_fsdfm = mdpo_fsdfm.spawn()

    logger.info("All 6 jobs submitted. Waiting for completion...")

    # Wait for all to finish
    h_espo_ref.get()
    logger.info("ESPO ReFusion mu=8 -- DONE")
    h_espo_fsdfm.get()
    logger.info("ESPO FS-DFM mu=8 -- DONE")
    h_cjgrpo_ref.get()
    logger.info("CJ-GRPO ReFusion -- DONE")
    h_cjgrpo_fsdfm.get()
    logger.info("CJ-GRPO FS-DFM -- DONE")
    h_mdpo_ref.get()
    logger.info("MDPO ReFusion -- DONE")
    h_mdpo_fsdfm.get()
    logger.info("MDPO FS-DFM -- DONE")

    logger.info("=" * w)
    logger.info("All 6 training jobs complete.")
    logger.info("Checkpoints on volume 'openbrowser-checkpoints' at:")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/espo-refusion/")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/espo-fsdfm/")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/cjgrpo-refusion/")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/cjgrpo-fsdfm/")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/mdpo-refusion/")
    logger.info("  /mnt/user_storage/openbrowser/checkpoints/mdpo-fsdfm/")
    logger.info("=" * w)


@app.local_entrypoint(name="eval_all")
def eval_all():
    """Run all 12 eval jobs (6 checkpoints x val+test).

    Usage: modal run infra/training/modal/openbrowser_modal.py::eval_all
    """
    w = 70
    logger.info("=" * w)
    logger.info("OpenBrowser Eval -- Modal Edition (6 checkpoints x val+test)")
    logger.info("  ESPO mu=8: ReFusion (L40S) + FS-DFM (A10)")
    logger.info("  CJ-GRPO:   ReFusion (L40S) + FS-DFM (A10)")
    logger.info("  MDPO:      ReFusion (L40S) + FS-DFM (A10)")
    logger.info("=" * w)

    # Spawn all 6 eval jobs in parallel (each runs val+test sequentially)
    handles = {
        "ESPO ReFusion mu=8": eval_espo_refusion_mu8.spawn(),
        "ESPO FS-DFM mu=8": eval_espo_fsdfm_mu8.spawn(),
        "CJ-GRPO ReFusion": eval_cjgrpo_refusion.spawn(),
        "CJ-GRPO FS-DFM": eval_cjgrpo_fsdfm.spawn(),
        "MDPO ReFusion": eval_mdpo_refusion.spawn(),
        "MDPO FS-DFM": eval_mdpo_fsdfm.spawn(),
    }

    logger.info("All 6 eval jobs submitted. Waiting for completion...")

    for name, handle in handles.items():
        handle.get()
        logger.info("%s eval -- DONE", name)

    logger.info("=" * w)
    logger.info("All 12 evaluations complete.")
    logger.info("Results on volume 'openbrowser-checkpoints' at:")
    logger.info("  /mnt/user_storage/openbrowser/eval/<method>-<model>-<split>/results.json")
    logger.info("=" * w)


# ===========================================================================
# PUSH CHECKPOINTS TO HUGGINGFACE
# ===========================================================================

HF_ORG = "billyenrizky"

# New checkpoints to push (sequence-level results)
HF_PUSH_MODELS = [
    (
        "espo-refusion",
        "ReFusion-8B-ESPO-mu8",
        "ReFusion 8B trained with ESPO mu=8 (ELBO-based Sequence-level Policy Optimization). "
        "Achieves 83.1% nonzero rate / 0.394 average reward on 124 test tasks (+22.6pp over SFT). "
        "Sequence-level RL with multi-epoch training (mu=8) and PPO clipping.",
    ),
    (
        "cjgrpo-refusion",
        "ReFusion-8B-CJ-GRPO",
        "ReFusion 8B trained with CJ-GRPO (Consistency-Justified GRPO). "
        "Achieves 83.9% nonzero rate / 0.390 average reward on 124 test tasks (+23.4pp over SFT). "
        "Per-step trajectory consistency with mu=1.",
    ),
    (
        "mdpo-refusion",
        "ReFusion-8B-MDPO",
        "ReFusion 8B trained with MDPO (Masked Diffusion Policy Optimization). "
        "Best result in the paper: 91.9% nonzero rate / 0.445 average reward on 124 test tasks "
        "(+31.4pp over SFT). Temporal advantage decomposition with mu=1.",
    ),
    (
        "espo-fsdfm",
        "FS-DFM-1.3B-ESPO-mu8",
        "FS-DFM 1.3B trained with ESPO mu=8 (ELBO-based Sequence-level Policy Optimization). "
        "First RL method to improve FS-DFM over SFT: 87.1% nonzero rate / 0.198 average reward "
        "on 124 test tasks (+18.6pp over SFT). Only ELBO-based methods generalize to DFM architectures.",
    ),
]


def _create_model_card(repo_name: str, description: str) -> str:
    """Generate a HuggingFace model card for sequence-level checkpoints."""
    return f"""---
tags:
  - discrete-flow-matching
  - web-action-planning
  - formfactory
  - reinforcement-learning
  - openbrowser
  - sequence-level-rl
license: apache-2.0
---

# {repo_name}

{description}

## Paper

**Generative Action Planning via Discrete Flow Matching with Online Reinforcement Fine-Tuning**
- Author: Muhammad Enrizky Brillian
- Institution: University of Toronto Scarborough
- Code: https://github.com/billy-enrizky/openbrowser-ai

## Training Details

- **Dataset**: FormFactory (992 train / 124 val / 124 test tasks, 25 form types, 8 domains)
- **Infrastructure**: NVIDIA L40S (ReFusion) / A10G (FS-DFM) on Modal.com
- **Framework**: PyTorch + PEFT (LoRA/QLoRA)
- **Training prompts**: 50 (sequence-level), G=4 rollouts per prompt

## Citation

```bibtex
@article{{brillian2026flowgrpo,
  title={{Generative Action Planning via Discrete Flow Matching with Online Reinforcement Fine-Tuning}},
  author={{Brillian, Muhammad Enrizky}},
  year={{2026}}
}}
```
"""


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[secret],
    timeout=1800,
    cpu=2,
)
def push_to_hf():
    """Push sequence-level RL checkpoints to HuggingFace.

    Usage: modal run infra/training/modal/openbrowser_modal.py::push_to_hf
    """
    from pathlib import Path
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set in openbrowser-secrets")
        return

    api = HfApi(token=token)
    ckpt_base = Path(f"{VOLUME_MOUNT}/openbrowser/checkpoints")

    available = [d.name for d in ckpt_base.iterdir() if d.is_dir()]
    logger.info("Available checkpoints: %s", available)

    success = 0
    for ckpt_name, repo_suffix, description in HF_PUSH_MODELS:
        ckpt_path = ckpt_base / ckpt_name
        repo_id = f"{HF_ORG}/{repo_suffix}"

        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s -- skipping %s", ckpt_path, repo_id)
            continue

        files = [f for f in ckpt_path.rglob("*") if f.is_file()]
        if not files:
            logger.warning("Checkpoint empty: %s -- skipping %s", ckpt_path, repo_id)
            continue

        logger.info("Pushing %s (%d files) -> %s", ckpt_path, len(files), repo_id)

        # Write model card
        card = _create_model_card(repo_suffix, description)
        (ckpt_path / "README.md").write_text(card)

        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(ckpt_path),
            repo_id=repo_id,
            commit_message=f"Upload {repo_suffix} checkpoint",
        )
        logger.info("Pushed %s -- https://huggingface.co/%s", repo_id, repo_id)
        success += 1

    logger.info("Done: %d/%d models pushed to HuggingFace", success, len(HF_PUSH_MODELS))
