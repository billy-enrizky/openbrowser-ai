"""Evaluate FS-DFM SFT-only checkpoint on FormFactory via browser execution.

Loads the FS-DFM LoRA checkpoint (no GRPO), generates action plans via
discrete Euler solver with prefix conditioning, executes them in a headless
browser, and computes rewards. Produces the same metrics as the GRPO trainer
(avg_reward, nonzero rate) for direct SFT vs GRPO comparison (H1).

Usage:
    FSDFM_SFT_CHECKPOINT=/mnt/user_storage/openbrowser/checkpoints/fsdfm-sft/lora_adapter \
    uv run infra/training/flow_matching/eval_fsdfm_sft.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
    ONLINE_FSDFM_GRPO_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    PolynomialConvexScheduler,
    generate_with_prefix_conditioning,
    inject_lora,
    load_fsdfm_from_huggingface,
    load_lora_weights,
)
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.utils import resolve_data_path

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
    logger.info(f"Loaded {len(records)} prompts for FS-DFM SFT evaluation")
    return records


async def evaluate():
    """Run FS-DFM SFT-only evaluation with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    grpo_config = ONLINE_FSDFM_GRPO_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if grpo_config.get("bf16") else torch.float16

    # Load GPT-2 tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flow matching scheduler
    exponent = model_config.get("scheduler_exponent", 2.0)
    scheduler = PolynomialConvexScheduler(exponent=exponent)

    # Load model with LoRA
    logger.info("Loading FS-DFM 1.3B model")
    model = load_fsdfm_from_huggingface(model_config, device=device, dtype=compute_dtype)
    model = inject_lora(model, model_config)

    # Load SFT checkpoint
    sft_checkpoint = os.environ.get("FSDFM_SFT_CHECKPOINT", "")
    if sft_checkpoint and Path(sft_checkpoint).exists():
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            logger.info(f"Loading SFT checkpoint: {lora_path}")
            load_lora_weights(model, str(lora_path))
        else:
            logger.warning(f"lora_weights.pt not found at {lora_path}")
    else:
        logger.error(f"SFT checkpoint not found: {sft_checkpoint}")
        return

    model.eval()

    # Load prompts
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    max_samples = int(os.environ.get("MAX_EVAL_SAMPLES", DATA_CONFIG.get("max_train_samples", 0)))
    prompts = load_prompts(train_file, max_samples=max_samples)

    max_seq_length = model_config["max_seq_length"]
    num_sampling_steps = grpo_config.get("num_sampling_steps", model_config.get("num_sampling_steps", 64))
    gen_temperature = model_config.get("generation_temperature", 1.0)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)

    # Start FormFactory server
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = grpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    # Start browser environment
    headless = grpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        f"Starting FS-DFM SFT evaluation: {len(prompts)} prompts, "
        f"sampling_steps={num_sampling_steps}, temperature={gen_temperature}"
    )

    all_rewards = []
    results = []

    try:
        for i, prompt_data in enumerate(prompts):
            # Periodic browser restart
            if i > 0 and i % 10 == 0:
                logger.info(f"Periodic browser restart (prompt {i})")
                await browser_env.close()
                browser_env = await BrowserEnvironment.create(headless=headless)

            instruction = prompt_data.get("instruction", prompt_data.get("condition", ""))
            form_url = prompt_data.get("url", "")
            ground_truth_fields = prompt_data.get("ground_truth_fields", {})

            if not instruction or not form_url:
                logger.warning(f"Skipping prompt {i}: missing instruction or url")
                continue

            # Tokenize instruction
            inst_enc = tokenizer(
                instruction,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length // 2,
                return_tensors="pt",
            )
            prefix_ids = inst_enc["input_ids"].to(device)
            prefix_len = prefix_ids.shape[1]
            gen_length = max(1, max_seq_length - prefix_len)

            # Generate via discrete Euler solver (greedy: temperature=0)
            generated_ids = generate_with_prefix_conditioning(
                model=model,
                prefix_ids=prefix_ids,
                gen_length=gen_length,
                config={**model_config, "num_sampling_steps": num_sampling_steps},
                scheduler=scheduler,
                temperature=0.0,  # Greedy for deterministic eval
            )
            response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Navigate to form
            await browser_env.reset()
            try:
                await browser_env.tools.navigate(
                    url=form_url,
                    new_tab=False,
                    browser_session=browser_env.browser_session,
                )
                await asyncio.sleep(0.5)
                element_map = await browser_env.get_element_map()
            except Exception as e:
                logger.warning(f"Navigation failed for prompt {i}: {e}")
                all_rewards.append(0.0)
                results.append({"prompt_idx": i, "reward": 0.0, "error": str(e)})
                continue

            # Parse and execute actions
            actions = parse_rollout_to_actions(response_text, element_map)
            if not actions:
                logger.info(f"Prompt {i}: no valid actions parsed")
                all_rewards.append(0.0)
                results.append({"prompt_idx": i, "reward": 0.0, "actions": 0})
                continue

            outcome = await browser_env.execute_actions(
                actions, timeout_per_action=action_timeout
            )
            reward = compute_online_reward(
                outcome,
                ground_truth_fields,
                weights=grpo_config.get("reward_weights"),
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
            logger.info("FS-DFM SFT-ONLY EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"  Prompts evaluated: {len(all_rewards)}")
            logger.info(f"  Nonzero reward rate: {nonzero}/{len(all_rewards)} ({100*nonzero/len(all_rewards):.1f}%)")
            logger.info(f"  Average reward: {avg_reward:.4f}")
            logger.info("=" * 60)

            # Save results
            output_dir = Path("outputs/eval_fsdfm_sft")
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "results.json"
            summary = {
                "model": "FS-DFM 1.3B SFT-only (LoRA)",
                "checkpoint": sft_checkpoint,
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
                dest = anyscale_storage / "fsdfm-sft-only"
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
