"""Evaluate a QLoRA checkpoint on FormFactory via greedy browser execution.

Loads a QLoRA checkpoint (SFT or GRPO), generates action plans via greedy
decoding, executes them in a headless browser against FormFactory forms,
and computes rewards. Produces nonzero rate and avg_reward for fair
apples-to-apples comparison across checkpoints.

Env vars:
    SFT_CHECKPOINT_PATH  Path to QLoRA adapter checkpoint
    EVAL_LABEL           Label for logs/output dir (default: "SFT-ONLY")
    MAX_EVAL_SAMPLES     Number of prompts to evaluate (default: from config)

Usage:
    SFT_CHECKPOINT_PATH=/mnt/user_storage/openbrowser/checkpoints/sft \
    uv run infra/training/finetuning/eval_sft.py

    SFT_CHECKPOINT_PATH=/mnt/user_storage/openbrowser/checkpoints/online-grpo \
    EVAL_LABEL=GRPO \
    uv run infra/training/finetuning/eval_sft.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.finetuning.config import DATA_CONFIG, ONLINE_GRPO_CONFIG
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.utils import format_chat_prompt, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for evaluation."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for SFT evaluation")
    return records


def load_sft_model(config: dict):
    """Load the SFT checkpoint with QLoRA quantization."""
    sft_checkpoint = config["sft_checkpoint"]
    if not sft_checkpoint or not Path(sft_checkpoint).exists():
        logger.error(
            f"SFT checkpoint not found at '{sft_checkpoint}'. "
            "Set SFT_CHECKPOINT_PATH env var."
        )
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_checkpoint}")

    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )

    # Load base model
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

    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Load SFT LoRA adapter
    logger.info(f"Loading SFT adapter from: {sft_checkpoint}")
    model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    model.eval()
    model.config.use_cache = True

    return model


def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 512
) -> str:
    """Generate a single response (greedy) for evaluation."""
    # Prepend empty think block to suppress Qwen3 thinking mode
    prompt_with_skip = prompt + "<think>\n</think>\n"
    inputs = tokenizer(prompt_with_skip, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for deterministic eval
            return_dict_in_generate=True,
            output_scores=False,
        )

    prompt_length = inputs.input_ids.shape[1]
    response_ids = outputs.sequences[0][prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


async def evaluate():
    """Run greedy evaluation with browser execution on FormFactory."""
    config = ONLINE_GRPO_CONFIG
    eval_label = os.environ.get("EVAL_LABEL", "SFT-ONLY")

    seed = int(os.environ.get("RANDOM_SEED", "42"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model (inference only)
    model = load_sft_model(config)

    # Load prompts from val or test split
    eval_split = os.environ.get("EVAL_SPLIT", "val")
    split_to_key = {"test": "test_file", "train": "train_file", "val": "val_file"}
    eval_file_key = split_to_key.get(eval_split, "val_file")
    eval_file = resolve_data_path(DATA_CONFIG[eval_file_key])
    max_samples = int(os.environ.get("MAX_EVAL_SAMPLES", DATA_CONFIG.get("max_eval_samples", 0)))
    prompts = load_prompts(eval_file, max_samples=max_samples)
    logger.info("Evaluating on %s split (%s)", eval_split, eval_file)

    max_new_tokens = config.get("max_new_tokens", 512)
    action_timeout = config.get("action_timeout_s", 10.0)

    # Start FormFactory server
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    # Start browser environment
    headless = config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(f"Starting {eval_label} evaluation: {len(prompts)} prompts")

    all_rewards = []
    results = []

    try:
        for i, prompt_data in enumerate(prompts):
            # Periodic browser restart to reset DOM element indices
            if i > 0 and i % 10 == 0:
                logger.info(f"Periodic browser restart (prompt {i}) to reset DOM indices")
                await browser_env.close()
                browser_env = await BrowserEnvironment.create(headless=headless)

            instruction = prompt_data.get("instruction", "")
            form_url = prompt_data.get("url", "")
            ground_truth_fields = prompt_data.get("ground_truth_fields", {})

            if not instruction or not form_url:
                logger.warning(f"Skipping prompt {i}: missing instruction or url")
                continue

            prompt_text = format_chat_prompt(instruction)

            # Generate single response (greedy)
            response = generate_response(
                model, tokenizer, prompt_text,
                max_new_tokens=max_new_tokens,
            )

            # Navigate to form
            await browser_env.reset()
            try:
                await browser_env.tools.navigate(
                    url=form_url,
                    new_tab=False,
                    browser_session=browser_env.browser_session,
                )
                await asyncio.sleep(0.5)
                await browser_env.bypass_html5_validation()
                element_map = await browser_env.get_element_map()
            except Exception as e:
                logger.warning(f"Navigation failed for prompt {i}: {e}")
                all_rewards.append(0.0)
                results.append({"prompt_idx": i, "reward": 0.0, "error": str(e)})
                continue

            # Parse response into actions
            actions = parse_rollout_to_actions(response, element_map)

            if not actions:
                logger.info(f"Prompt {i}: no valid actions parsed")
                all_rewards.append(0.0)
                results.append({"prompt_idx": i, "reward": 0.0, "actions": 0})
                continue

            # Execute actions in browser
            outcome = await browser_env.execute_actions(
                actions, timeout_per_action=action_timeout
            )

            # Compute reward
            reward = compute_online_reward(
                outcome,
                ground_truth_fields,
                weights=config.get("reward_weights"),
            )

            all_rewards.append(reward)
            results.append({
                "prompt_idx": i,
                "reward": reward,
                "actions": len(actions),
                "executed": outcome.actions_executed,
                "success_page": outcome.success_page_detected,
            })

            logger.info(
                f"  Prompt {i+1}/{len(prompts)}: reward={reward:.3f}, "
                f"actions={len(actions)}, executed={outcome.actions_executed}, "
                f"success={outcome.success_page_detected}"
            )

        # Final summary
        if all_rewards:
            avg_reward = sum(all_rewards) / len(all_rewards)
            nonzero = sum(1 for r in all_rewards if r > 0)
            logger.info("=" * 60)
            logger.info(f"{eval_label} EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"  Prompts evaluated: {len(all_rewards)}")
            logger.info(f"  Nonzero reward rate: {nonzero}/{len(all_rewards)} ({100*nonzero/len(all_rewards):.1f}%)")
            logger.info(f"  Average reward: {avg_reward:.4f}")
            logger.info("=" * 60)

            # Save results to JSON
            output_dir = Path(f"outputs/eval_{eval_label.lower().replace(' ', '_')}")
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "results.json"
            summary = {
                "model": f"Qwen3-8B {eval_label} (QLoRA)",
                "checkpoint": config["sft_checkpoint"],
                "prompts_evaluated": len(all_rewards),
                "nonzero_count": nonzero,
                "nonzero_rate": nonzero / len(all_rewards),
                "avg_reward": avg_reward,
                "per_prompt": results,
            }
            with open(results_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Results saved to {results_file}")

            # Persist to Anyscale storage
            anyscale_storage = Path("/mnt/user_storage/openbrowser/eval")
            if anyscale_storage.parent.parent.exists():
                dest = anyscale_storage / eval_label.lower().replace(" ", "-")
                dest.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(str(results_file), str(dest / "results.json"))
                logger.info(f"Results persisted to {dest}")
        else:
            logger.warning("No prompts evaluated")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(evaluate())
