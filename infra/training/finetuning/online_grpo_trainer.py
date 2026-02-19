"""Online GRPO trainer: AR policy with real browser execution on FormFactory.

Unlike the offline grpo_trainer.py which scores rollouts via text-matching
heuristics, this trainer executes generated action plans in a real browser
(via the openbrowser package) against a running FormFactory Flask server
and computes rewards from actual form submission outcomes.

Architecture:
    1. Qwen3-8B QLoRA model generates G candidate plans via autoregressive sampling
    2. Each plan is parsed into executable actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = actual form submission success + field accuracy
    5. REINFORCE with group-relative advantages + non-negative KL penalty (Schulman k3)

Usage:
    SFT_CHECKPOINT_PATH=outputs/finetuning_sft/final uv run infra/training/finetuning/online_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.finetuning.config import DATA_CONFIG, ONLINE_GRPO_CONFIG
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import (
    format_chat_prompt,
    resolve_data_path,
    persist_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_prompts(file_path: str, max_samples: int = 0, shuffle: bool = True) -> list[dict]:
    """Load prompts for GRPO rollouts.

    Args:
        file_path: Path to JSONL file with training prompts.
        max_samples: Truncate to this many samples (0 = use all).
        shuffle: Shuffle prompts to avoid long blocks of identical form types.
            The raw FormFactory data has 40 consecutive samples per form type,
            which causes GRPO to skip most steps (zero reward variance).
    """
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if shuffle:
        random.seed(42)
        random.shuffle(records)
        logger.info("Shuffled training prompts to break sequential form-type blocks")
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for online GRPO")
    return records


def load_quantized_model(model_name: str, config: dict):
    """Load a model with 4-bit quantization."""
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    # Pre-quantized models (e.g. unsloth/Qwen3-8B-bnb-4bit) already have
    # quantization_config embedded -- passing it again triggers a warning.
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
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


def generate_rollouts(
    model, tokenizer, prompt: str, group_size: int, max_new_tokens: int = 512,
    temperature: float = 1.0,
) -> tuple[list[str], torch.Tensor, int]:
    """Generate G rollouts for a single prompt.

    Returns:
        responses: list of decoded response strings
        all_input_ids: tensor of full sequences [G, seq_len]
        prompt_length: length of the prompt tokens
    """
    # Prepend empty think block to suppress Qwen3 thinking mode --
    # model sees <think>\n</think>\n as already completed and generates actions directly
    prompt_with_skip = prompt + "<think>\n</think>\n"
    inputs = tokenizer(prompt_with_skip, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]

    # Enable KV cache for generation (disabled during training for gradient checkpointing)
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=False,
        )
    model.config.use_cache = False
    model.train()

    all_sequences = outputs.sequences  # [G, total_len]

    responses = []
    for seq in all_sequences:
        text = tokenizer.decode(
            seq[prompt_length:], skip_special_tokens=True
        )
        responses.append(text)

    return responses, all_sequences, prompt_length


def compute_per_token_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log probabilities for the response portion.

    Processes one sample at a time to avoid CUDA OOM from materializing
    [B, seq_len, vocab_size] tensors (Qwen3-8B has vocab=152064, which
    at B=4 and seq_len=1024 would require ~2.5GB for log_softmax alone).

    Uses F.cross_entropy which fuses log_softmax + gather internally
    and never materializes the full [seq_len, vocab] softmax tensor.

    Args:
        model: the language model
        input_ids: [B, seq_len] full sequences
        attention_mask: [B, seq_len] mask (1 for real tokens, 0 for padding)
        prompt_length: number of prompt tokens to skip

    Returns:
        token_log_probs: [B, max_resp_len] per-token log-probs (0 for padding)
        resp_mask: [B, max_resp_len] float mask (1.0 for valid response tokens)
    """
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    response_start = max(0, prompt_length - 1)  # -1 for shift offset
    max_resp_len = seq_len - 1 - response_start

    all_log_probs = []
    all_masks = []

    for i in range(B):
        ids_i = input_ids[i : i + 1]  # [1, seq_len]
        mask_i = attention_mask[i : i + 1]  # [1, seq_len]

        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=ids_i, attention_mask=mask_i)
            logits = outputs.logits  # [1, seq_len, vocab]

        # Shift: predict token t+1 from position t
        shift_logits = logits[0, :-1, :]  # [seq_len-1, vocab]
        shift_labels = ids_i[0, 1:]  # [seq_len-1]

        # Use cross_entropy with reduction='none' -- fuses log_softmax + gather
        # internally without materializing the full softmax tensor
        token_nll = F.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        )  # [seq_len-1]
        token_log_probs = -token_nll  # log_prob = -cross_entropy

        # Only take response tokens (skip prompt)
        response_log_probs = token_log_probs[response_start:]
        mask_shifted = mask_i[0, 1:][response_start:].float()

        all_log_probs.append(response_log_probs[:max_resp_len])
        all_masks.append(mask_shifted[:max_resp_len])

    return torch.stack(all_log_probs), torch.stack(all_masks)


async def train():
    """Run online GRPO training loop with browser execution."""
    config = ONLINE_GRPO_CONFIG

    # Determine model to load: SFT checkpoint or base model
    sft_checkpoint = config["sft_checkpoint"]
    if sft_checkpoint and Path(sft_checkpoint).exists():
        logger.info(f"Loading SFT checkpoint from: {sft_checkpoint}")
        model_name = sft_checkpoint
        is_peft_checkpoint = True
    else:
        if sft_checkpoint:
            logger.warning(
                f"SFT checkpoint not found at {sft_checkpoint}, "
                "falling back to base model"
            )
        model_name = config["model_name"]
        is_peft_checkpoint = False

    logger.info(f"Loading tokenizer from: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model with QLoRA
    logger.info(f"Loading policy model: {model_name}")
    if is_peft_checkpoint:
        base_model = load_quantized_model(config["model_name"], config)
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)
        model.train()
    else:
        model = load_quantized_model(model_name, config)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load reference model (frozen, for KL computation)
    logger.info("Loading reference model (frozen)")
    ref_model = load_quantized_model(config["model_name"], config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_model, model_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load training data
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Optimizer -- only LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])

    group_size = config["group_size"]
    kl_coeff = config["kl_coeff"]
    max_new_tokens = config.get("max_new_tokens", 512)
    action_timeout = config.get("action_timeout_s", 5.0)
    temperature = config.get("temperature", 1.0)
    min_reward_variance = config.get("min_reward_variance", 0.01)

    # Start FormFactory server
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    # Start browser environment (openbrowser)
    headless = config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        f"Starting online GRPO training: {len(prompts)} prompts, G={group_size}, "
        f"kl_coeff={kl_coeff}"
    )

    total_steps = 0
    try:
        for epoch in range(config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
            epoch_rewards = []
            epoch_kl = []

            consecutive_zero_reward = 0

            for i, prompt_data in enumerate(prompts):
                # Periodically restart browser to reset DOM element indices.
                # DOMWatchdog assigns monotonically increasing indices across
                # navigations -- after ~10 forms, indices reach 19000+ which
                # slows CDP communication and causes action timeouts.
                if i > 0 and i % 10 == 0:
                    logger.info(f"Periodic browser restart (prompt {i}) to reset DOM indices")
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)

                # Health check: restart FormFactory server if it died
                if not ff_server.is_healthy():
                    logger.warning("FormFactory server is not responding, restarting...")
                    if not ff_server.restart():
                        logger.error("Failed to restart FormFactory server, aborting training")
                        break
                    # Also restart browser after server restart
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)
                    consecutive_zero_reward = 0

                instruction = prompt_data.get("instruction", "")
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})

                if not instruction or not form_url:
                    logger.warning(
                        f"Skipping prompt {i}: missing instruction or url"
                    )
                    continue

                prompt_text = format_chat_prompt(instruction)

                # Generate G rollouts
                model.eval()
                rollouts, sequences, prompt_length = generate_rollouts(
                    model, tokenizer, prompt_text, group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                model.train()

                # Pad sequences to same length for batched computation
                max_len = max(seq.shape[0] for seq in sequences)
                padded = torch.zeros(
                    group_size, max_len, dtype=torch.long, device=sequences.device
                )
                attention_mask = torch.zeros(
                    group_size, max_len, dtype=torch.long, device=sequences.device
                )
                for j, seq in enumerate(sequences):
                    padded[j, : seq.shape[0]] = seq
                    attention_mask[j, : seq.shape[0]] = 1

                # Execute each rollout in browser and score
                rewards = []
                for g, rollout_text in enumerate(rollouts):
                    # Reset browser and navigate to form page
                    await browser_env.reset()

                    try:
                        await browser_env.tools.navigate(
                            url=form_url,
                            new_tab=False,
                            browser_session=browser_env.browser_session,
                        )
                        # Brief wait for page to load
                        await asyncio.sleep(0.5)
                        element_map = await browser_env.get_element_map()
                    except Exception as e:
                        logger.warning(f"Navigation failed for rollout {g}: {e}")
                        rewards.append(0.0)
                        continue

                    # Parse rollout text into executable actions
                    actions = parse_rollout_to_actions(rollout_text, element_map)

                    if not actions:
                        logger.debug(f"No valid actions parsed from rollout {g}")
                        rewards.append(0.0)
                        continue

                    # Execute actions in browser
                    outcome = await browser_env.execute_actions(
                        actions, timeout_per_action=action_timeout
                    )

                    # Compute reward from browser outcome
                    reward = compute_online_reward(
                        outcome,
                        ground_truth_fields,
                        weights=config.get("reward_weights"),
                    )
                    rewards.append(reward)

                epoch_rewards.extend(rewards)

                # Skip gradient update when all rollouts failed or reward
                # variance is too low.  Without reward variance, GRPO
                # advantages are all zero and the only gradient signal is
                # the KL penalty, which pushes the policy back toward
                # the reference model and degrades performance.
                all_zero = all(r == 0.0 for r in rewards)
                reward_mean = sum(rewards) / len(rewards) if rewards else 0
                reward_var = (
                    sum((r - reward_mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
                    if len(rewards) > 1 else 0.0
                )
                low_variance = reward_var < min_reward_variance

                if all_zero or low_variance:
                    consecutive_zero_reward += 1
                    total_steps += 1
                    skip_reason = "all-zero" if all_zero else f"low-var={reward_var:.6f}"
                    if total_steps % config["logging_steps"] == 0:
                        logger.info(
                            f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                            f"avg_reward={reward_mean:.3f} [SKIPPED: {skip_reason}, "
                            f"{consecutive_zero_reward} consecutive]"
                        )
                    # If too many consecutive skips, force browser restart
                    if consecutive_zero_reward >= 3 and ff_server.is_healthy():
                        logger.warning(
                            f"{consecutive_zero_reward} consecutive skipped prompts "
                            "despite healthy server -- restarting browser"
                        )
                        await browser_env.close()
                        browser_env = await BrowserEnvironment.create(headless=headless)
                        consecutive_zero_reward = 0
                    continue
                else:
                    consecutive_zero_reward = 0

                # Compute group-relative advantages
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=padded.device
                )

                # Compute per-token policy log-probs (with gradients)
                policy_token_lp, resp_mask = compute_per_token_log_probs(
                    model, padded, attention_mask, prompt_length
                )  # [G, T], [G, T]

                # Compute per-token reference log-probs (for KL, no gradients)
                with torch.no_grad():
                    ref_token_lp, _ = compute_per_token_log_probs(
                        ref_model, padded, attention_mask, prompt_length
                    )  # [G, T]

                # Per-sample mean log-prob under current policy
                tokens_per_sample = resp_mask.sum(dim=-1).clamp(min=1)  # [G]
                sample_log_prob = (
                    (policy_token_lp * resp_mask).sum(dim=-1) / tokens_per_sample
                )  # [G]

                # REINFORCE with group-relative advantages
                pg_loss = -(advantages_t * sample_log_prob).mean()

                # KL divergence: Schulman k3 approximation (always >= 0)
                # D_KL(pi || ref) ~ r - log(r) - 1  where r = pi_ref / pi_theta
                log_r = ref_token_lp - policy_token_lp  # [G, T]
                r = torch.exp(log_r)
                kl_per_token = r - log_r - 1  # >= 0 by Jensen's inequality
                total_resp_tokens = resp_mask.sum().clamp(min=1)
                kl_div = (kl_per_token * resp_mask).sum() / total_resp_tokens

                # KL penalty (kl_div >= 0, so this always penalizes divergence)
                kl_penalty = kl_coeff * kl_div

                # Total loss
                loss = pg_loss + kl_penalty

                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = kl_div.item()
                epoch_kl.append(avg_kl)

                if total_steps % config["logging_steps"] == 0:
                    logger.info(
                        f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                        f"avg_reward={avg_reward:.3f}, "
                        f"loss={loss.item():.4f}, "
                        f"pg_loss={pg_loss.item():.4f}, "
                        f"kl={avg_kl:.4f}"
                    )

                if config["save_steps"] > 0 and total_steps % config["save_steps"] == 0:
                    ckpt_dir = f"outputs/finetuning_online_grpo/checkpoint-{total_steps}"
                    model.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    f"Epoch {epoch + 1} complete: avg_reward={epoch_avg:.3f}, "
                    f"nonzero_rewards={nonzero}/{len(epoch_rewards)}, "
                    f"avg_kl={sum(epoch_kl)/len(epoch_kl):.4f}"
                )

        # Save final model
        final_dir = "outputs/finetuning_online_grpo/final"
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Online GRPO training complete. Model saved to {final_dir}")

        # Persist checkpoint to Anyscale storage
        persist_checkpoint(final_dir, "online-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
