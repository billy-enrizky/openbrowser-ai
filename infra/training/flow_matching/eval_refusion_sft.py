"""Evaluate ReFusion SFT-only checkpoint on FormFactory via browser execution.

Loads the ReFusion QLoRA SFT checkpoint (no GRPO), generates action plans
via iterative unmasking (masked diffusion), executes them in a headless
browser, and computes rewards. Produces the same metrics as the GRPO trainer
(avg_reward, nonzero rate) for direct SFT vs GRPO comparison (H1).

Usage:
    FLOW_LLM_SFT_CHECKPOINT=/mnt/user_storage/openbrowser/checkpoints/flow-llm-sft/adapter \
    uv run infra/training/flow_matching/eval_refusion_sft.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
    ONLINE_FLOW_GRPO_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import FlowLLM
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
    logger.info(f"Loaded {len(records)} prompts for ReFusion SFT evaluation")
    return records


def load_refusion_sft(config: dict):
    """Load ReFusion with QLoRA + SFT LoRA adapter."""
    sft_checkpoint = os.environ.get(
        "FLOW_LLM_SFT_CHECKPOINT",
        "/mnt/user_storage/openbrowser/checkpoints/flow-llm-sft/adapter",
    )
    model_name = config["base_model_name"]

    compute_dtype = (
        torch.bfloat16
        if config.get("bnb_4bit_compute_dtype") == "bfloat16"
        else torch.float16
    )

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info(f"Loading ReFusion base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=config.get("trust_remote_code", True),
    )

    # Load SFT adapter
    if Path(sft_checkpoint).exists():
        logger.info(f"Loading SFT adapter from: {sft_checkpoint}")
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    else:
        logger.error(f"SFT checkpoint not found: {sft_checkpoint}")
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_checkpoint}")

    model.eval()

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


async def evaluate():
    """Run ReFusion SFT-only evaluation with browser execution."""
    llm_config = FLOW_LLM_CONFIG
    grpo_config = ONLINE_FLOW_GRPO_CONFIG

    # Load model + tokenizer
    model, tokenizer = load_refusion_sft(llm_config)
    mask_token_id = llm_config.get("mask_token_id", 151670)
    flow_llm = FlowLLM(model, tokenizer, mask_token_id=mask_token_id)

    # Load prompts from val or test split
    eval_split = os.environ.get("EVAL_SPLIT", "val")
    eval_file_key = "test_file" if eval_split == "test" else "val_file"
    eval_file = resolve_data_path(DATA_CONFIG[eval_file_key])
    max_samples = int(os.environ.get("MAX_EVAL_SAMPLES", DATA_CONFIG.get("max_eval_samples", 0)))
    prompts = load_prompts(eval_file, max_samples=max_samples)
    logger.info("Evaluating on %s split (%s)", eval_split, eval_file)

    max_new_tokens = llm_config.get("max_new_tokens", 512)
    num_denoising_steps = llm_config.get("num_denoising_steps", 64)
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
        f"Starting ReFusion SFT evaluation: {len(prompts)} prompts, "
        f"denoising_steps={num_denoising_steps}"
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

            # Format as ChatML prompt
            prompt_text = format_chat_prompt(instruction)

            # Tokenize condition
            cond_enc = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            condition_ids = cond_enc["input_ids"].to(flow_llm.device)
            condition_mask = cond_enc["attention_mask"].to(flow_llm.device)

            # Generate via iterative unmasking (temperature=0 for greedy eval)
            generated_ids = flow_llm.generate(
                condition_ids=condition_ids,
                condition_mask=condition_mask,
                seq_length=max_new_tokens,
                num_steps=num_denoising_steps,
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
            logger.info("REFUSION SFT-ONLY EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"  Prompts evaluated: {len(all_rewards)}")
            logger.info(f"  Nonzero reward rate: {nonzero}/{len(all_rewards)} ({100*nonzero/len(all_rewards):.1f}%)")
            logger.info(f"  Average reward: {avg_reward:.4f}")
            logger.info("=" * 60)

            # Save results
            output_dir = Path("outputs/eval_refusion_sft")
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "results.json"
            summary = {
                "model": "ReFusion 8B SFT-only (QLoRA)",
                "checkpoint": os.environ.get("FLOW_LLM_SFT_CHECKPOINT", ""),
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
                dest = anyscale_storage / "refusion-sft-only"
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
