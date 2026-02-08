"""GRPO trainer: Group Relative Policy Optimization on Mind2Web."""

import json
import logging

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from infra.training.shared.reward_functions import compute_grpo_advantages, compute_reward
from infra.training.finetuning.config import GRPO_CONFIG, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for GRPO")
    return records


def generate_rollouts(
    model, tokenizer, prompt: str, group_size: int, max_length: int = 512
) -> list[str]:
    """Generate G rollouts for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_return_sequences=group_size,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    responses = []
    for output in outputs:
        text = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(text)
    return responses


def train():
    """Run GRPO training loop."""
    config = GRPO_CONFIG

    checkpoint = config.get("sft_checkpoint", "")
    model_name = checkpoint if checkpoint else config["model_name"]

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config["fp16"] else torch.float32,
        device_map="auto",
    )

    # Load reference model for KL
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ref_model.eval()

    prompts = load_prompts(DATA_CONFIG["train_file"], max_samples=DATA_CONFIG.get("max_train_samples", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    group_size = config["group_size"]

    logger.info(f"Starting GRPO training: {len(prompts)} prompts, G={group_size}")

    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")

        for i, prompt_data in enumerate(prompts):
            instruction = prompt_data.get("instruction", "")
            ground_truth = prompt_data.get("response", "")

            # Generate G rollouts
            rollouts = generate_rollouts(model, tokenizer, instruction, group_size)

            # Score each rollout
            rewards = []
            for rollout in rollouts:
                signal = compute_reward(
                    agent_output=rollout,
                    ground_truth=ground_truth,
                    success="step" in rollout.lower(),
                    steps_taken=rollout.count("\n") + 1,
                    weights=config.get("reward_weights"),
                )
                rewards.append(signal.total)

            # Compute advantages
            advantages = compute_grpo_advantages(rewards, group_size)

            # Policy gradient update (simplified)
            # Full implementation would compute log-probs and KL properly
            loss = torch.tensor(0.0, requires_grad=True)
            for rollout, advantage in zip(rollouts, advantages):
                if advantage > 0:
                    tokens = tokenizer(rollout, return_tensors="pt").to(model.device)
                    logits = model(**tokens).logits
                    # Simplified loss: weighted by advantage
                    log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
                    loss = loss - advantage * log_probs.mean()

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if (i + 1) % config["logging_steps"] == 0:
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                logger.info(
                    f"  Step {i+1}/{len(prompts)}: avg_reward={avg_reward:.3f}, "
                    f"loss={loss.item():.4f}"
                )

    model.save_pretrained("outputs/finetuning_grpo/final")
    tokenizer.save_pretrained("outputs/finetuning_grpo/final")
    logger.info("GRPO training complete")


if __name__ == "__main__":
    train()
