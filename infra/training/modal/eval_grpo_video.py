"""
Qwen3-8B GRPO Eval with Video Recording, Modal Deployment
============================================================

Runs the Qwen3-8B GRPO-LoRA checkpoint on FormFactory forms,
records browser video via CDP screencast, and saves results + mp4
to a Modal volume.

Pre-requisites
--------------
    modal secret create openbrowser-secrets HF_TOKEN=hf_xxx

Usage
-----
Run eval on tech-software forms (best GRPO domain) with video:

    modal run infra/training/modal/eval_grpo_video.py

Run on a specific form category:

    modal run infra/training/modal/eval_grpo_video.py --category tech-software

Run on a specific form:

    modal run infra/training/modal/eval_grpo_video.py --form tech-software/bug-report --max-prompts 1
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal
from modal import App, Image as ModalImage, Volume, Secret, FilePatternMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOLUME_MOUNT = "/mnt/user_storage"
EVAL_TIMEOUT = 3600  # 1 hour
HF_GRPO_REPO = "billyenrizky/Qwen3-8B-FormFactory-GRPO-LoRA"
GRPO_CKPT_PATH = f"{VOLUME_MOUNT}/openbrowser/checkpoints/online-grpo"

app = App("openbrowser-grpo-eval-video")
volume = Volume.from_name("openbrowser-checkpoints", create_if_missing=True)
secret = Secret.from_name("openbrowser-secrets")

image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install(
        "libnss3", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2",
        "libxkbcommon0", "libxcomposite1", "libxdamage1", "libxrandr2", "libgbm1",
        "libasound2t64", "libpango-1.0-0", "libpangocairo-1.0-0", "libgtk-3-0",
        "fonts-liberation", "xdg-utils", "wget", "git", "build-essential", "curl",
    )
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
        "openbrowser-ai[video]",
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
    .run_commands(
        "playwright install chromium",
        "CHROME_PATH=$(python -c \"from playwright.sync_api import sync_playwright; "
        "pw=sync_playwright().start(); print(pw.chromium.executable_path); pw.stop()\" "
        "2>/dev/null) && if [ -n \"$CHROME_PATH\" ] && [ -f \"$CHROME_PATH\" ]; "
        "then ln -sf \"$CHROME_PATH\" /usr/bin/chromium; fi",
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str):
    """Shell out to bash, raise on failure."""
    logger.info(">>> %s", cmd)
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {cmd}")


def _ensure_grpo_checkpoint():
    """Download GRPO LoRA from HuggingFace if not on volume."""
    if Path(GRPO_CKPT_PATH).exists() and any(Path(GRPO_CKPT_PATH).iterdir()):
        logger.info("GRPO checkpoint found on volume: %s", GRPO_CKPT_PATH)
        return
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    logger.info("Downloading GRPO LoRA from %s", HF_GRPO_REPO)
    os.makedirs(GRPO_CKPT_PATH, exist_ok=True)
    snapshot_download(repo_id=HF_GRPO_REPO, local_dir=GRPO_CKPT_PATH, token=token)
    volume.commit()
    logger.info("GRPO checkpoint downloaded to %s", GRPO_CKPT_PATH)


def _download_formfactory():
    """Download FormFactory dataset."""
    _run("python -m infra.eval.scripts.download_datasets --datasets formfactory")


def _load_prompts(data_file: str, category: str = "", form: str = "", max_prompts: int = 5):
    """Load filtered prompts from data JSONL."""
    records = []
    with open(data_file) as f:
        for line in f:
            d = json.loads(line)
            url = d.get("url", "")
            if form and form not in url:
                continue
            if category and not form and category not in url:
                continue
            records.append(d)
            if len(records) >= max_prompts:
                break
    logger.info("Loaded %d prompts (filter: category=%s, form=%s)", len(records), category, form)
    return records


def _collect_video(video_dir: Path, dest_path: Path) -> bool:
    """Find recorded mp4 in video_dir and move to dest_path.

    RecordingWatchdog writes a UUID-named .mp4 file into the
    record_video_dir. This finds it and renames to a meaningful name.
    """
    mp4s = list(video_dir.glob("*.mp4"))
    if not mp4s:
        logger.warning("No video file found in %s", video_dir)
        return False
    src = mp4s[0]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest_path))
    logger.info("Video saved: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    return True


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    secrets=[secret],
    timeout=EVAL_TIMEOUT,
)
def eval_grpo_with_video(
    category: str = "tech-software",
    form: str = "",
    max_prompts: int = 5,
    split: str = "val",
):
    """Evaluate Qwen3-8B GRPO on FormFactory forms with video recording.

    Runs eval on filtered prompts, records CDP screencast video for each,
    saves results + videos to the Modal volume.
    """
    volume.reload()
    _ensure_grpo_checkpoint()
    _download_formfactory()

    results = asyncio.run(_eval_async(category, form, max_prompts, split))

    # Save results to volume
    output_dir = Path(f"{VOLUME_MOUNT}/openbrowser/eval/grpo-video")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    # Print summary
    logger.info("=" * 60)
    logger.info("GRPO VIDEO EVAL RESULTS")
    logger.info("=" * 60)
    for r in results:
        status = "PERFECT" if r["reward"] >= 0.999 else f"reward={r['reward']:.3f}"
        logger.info("  %s: %s (video: %s)", r["form"], status, r.get("video_path", "none"))
    perfect = [r for r in results if r["reward"] >= 0.999]
    logger.info("Perfect scores: %d/%d", len(perfect), len(results))
    logger.info("Videos saved to: %s", output_dir)
    logger.info("=" * 60)

    return results


async def _eval_async(category: str, form: str, max_prompts: int, split: str):
    """Async eval with model loading, browser, and video recording."""
    import tempfile

    # Increase bubus TypeTextEvent timeout: screencast adds CDP overhead to typing
    os.environ["TIMEOUT_TypeTextEvent"] = "120"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from infra.training.finetuning.config import ONLINE_GRPO_CONFIG, DATA_CONFIG
    from infra.training.shared.action_parser import parse_rollout_to_actions
    from infra.training.shared.browser_env import BrowserEnvironment
    from infra.training.shared.formfactory_server import FormFactoryServer
    from infra.training.shared.online_reward import compute_online_reward
    from infra.training.shared.utils import format_chat_prompt, resolve_data_path

    config = ONLINE_GRPO_CONFIG

    # Load model
    logger.info("Loading Qwen3-8B + GRPO LoRA...")
    compute_dtype = torch.bfloat16
    model_name = config["model_name"]
    is_prequantized = "bnb" in model_name.lower()
    load_kwargs = {"device_map": "auto", "torch_dtype": compute_dtype}
    if not is_prequantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["load_in_4bit"],
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, GRPO_CKPT_PATH)
    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded")

    # Load prompts
    split_key = {"val": "val_file", "test": "test_file", "train": "train_file"}.get(split, "val_file")
    data_file = resolve_data_path(DATA_CONFIG[split_key])
    prompts = _load_prompts(data_file, category=category, form=form, max_prompts=max_prompts)
    if not prompts:
        logger.error("No prompts found for category=%s, form=%s", category, form)
        return []

    # Start FormFactory
    formfactory_dir = Path("/root/openbrowser/data/formfactory")
    ff_server = FormFactoryServer(formfactory_dir, port=5050)
    if not ff_server.start():
        logger.error("FormFactory server failed to start")
        return []

    results = []
    final_video_dir = Path(f"{VOLUME_MOUNT}/openbrowser/eval/grpo-video/videos")
    final_video_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, prompt_data in enumerate(prompts):
            instruction = prompt_data.get("instruction", "")
            form_url = prompt_data.get("url", "")
            ground_truth = prompt_data.get("ground_truth_fields", {})
            form_name = "/".join(form_url.rstrip("/").split("/")[-2:])

            logger.info("--- Prompt %d/%d: %s ---", i + 1, len(prompts), form_name)

            # Generate response
            prompt_text = format_chat_prompt(instruction)
            prompt_with_skip = prompt_text + "<think>\n</think>\n"
            inputs = tokenizer(prompt_with_skip, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            prompt_length = inputs.input_ids.shape[1]
            response_ids = outputs.sequences[0][prompt_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            logger.info("Generated response (%d tokens)", len(response_ids))

            # Per-prompt temp dir for video recording
            prompt_video_dir = tempfile.mkdtemp(prefix=f"grpo_video_{i}_")

            # Create browser with CDP screencast recording
            # Tall viewport (1280x1080) so full form + submit button visible
            browser_env = await BrowserEnvironment.create(
                headless=True,
                record_video_dir=prompt_video_dir,
                record_video_size={"width": 1280, "height": 1080},
                viewport={"width": 1280, "height": 1080},
            )

            video_dest = final_video_dir / f"{form_name.replace('/', '_')}_{i}.mp4"

            try:
                await browser_env.navigate_with_timeout(form_url, timeout=30.0)
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.warning("Navigation failed: %s", e)
                await browser_env.close()
                shutil.rmtree(prompt_video_dir, ignore_errors=True)
                results.append({"form": form_name, "reward": 0.0, "error": str(e)})
                continue

            # Bypass HTML5 validation and get element map
            await browser_env.bypass_html5_validation()
            element_map = await browser_env.get_element_map()

            # Parse and execute actions
            actions = parse_rollout_to_actions(response, element_map)
            logger.info("Parsed %d actions", len(actions))

            if actions:
                outcome = await browser_env.execute_actions(
                    actions, timeout_per_action=60.0,
                )
            else:
                from infra.training.shared.online_reward import BrowserOutcome
                outcome = BrowserOutcome(
                    success_page_detected=False,
                    submitted_values={},
                    actions_executed=0,
                    total_actions=0,
                )

            # Force repaints so screencast captures success page
            # (static pages don't trigger compositor frames)
            await asyncio.sleep(1.0)
            try:
                cdp_session = await browser_env.browser_session.get_or_create_cdp_session()
                for _ in range(30):
                    await cdp_session.cdp_client.send.Runtime.evaluate(
                        params={"expression": "window.scrollBy(0, 1); window.scrollBy(0, -1);"},
                        session_id=cdp_session.session_id,
                    )
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug("Repaint trigger ended: %s", e)
            await browser_env.close()

            # Move UUID video to meaningful filename
            _collect_video(Path(prompt_video_dir), video_dest)
            shutil.rmtree(prompt_video_dir, ignore_errors=True)

            # Compute reward
            reward = compute_online_reward(
                outcome, ground_truth,
                weights=config.get("reward_weights"),
            )

            results.append({
                "form": form_name,
                "prompt_idx": i,
                "reward": reward,
                "actions_parsed": len(actions),
                "actions_executed": outcome.actions_executed,
                "success_page": outcome.success_page_detected,
                "video_path": str(video_dest),
                "response": response[:500],
            })

            logger.info(
                "Prompt %d: reward=%.3f, actions=%d/%d, success=%s, video=%s",
                i + 1, reward, outcome.actions_executed, len(actions),
                outcome.success_page_detected, video_dest,
            )

    finally:
        ff_server.stop()

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    category: str = "tech-software",
    form: str = "",
    max_prompts: int = 5,
    split: str = "val",
):
    """Run GRPO eval with video recording.

    Defaults to tech-software category (highest GRPO improvement domain).
    """
    results = eval_grpo_with_video.remote(
        category=category,
        form=form,
        max_prompts=max_prompts,
        split=split,
    )

    # Print results locally
    print("\n" + "=" * 60)
    print("GRPO VIDEO EVAL RESULTS")
    print("=" * 60)
    perfect_forms = []
    for r in results:
        status = "PERFECT (1.000)" if r["reward"] >= 0.999 else f"reward={r['reward']:.3f}"
        print(f"  {r['form']}: {status}")
        if r["reward"] >= 0.999:
            perfect_forms.append(r)
    print(f"\nPerfect scores: {len(perfect_forms)}/{len(results)}")
    print(f"Videos on volume: {VOLUME_MOUNT}/openbrowser/eval/grpo-video/videos/")
    print("=" * 60)

    if perfect_forms:
        print(f"\nTo download a perfect-score video:")
        vid = perfect_forms[0]["video_path"]
        fname = Path(vid).name
        print(f"  modal volume get openbrowser-checkpoints openbrowser/eval/grpo-video/videos/{fname} ./{fname}")
