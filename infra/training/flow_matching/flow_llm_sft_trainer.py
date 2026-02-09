"""Flow LLM SFT trainer: Masked diffusion denoising on ReFusion.

Trains ReFusion (GSAI-ML/ReFusion) with QLoRA for form-filling action plan
generation via masked diffusion. ReFusion's forward() handles the training
objective natively:
    1. Splits response into random-size slots (4/8/16/32 tokens)
    2. Randomly masks some slots, keeps others in shuffled order
    3. Computes hybrid loss: AR on unmasked + MDM on masked (normalized by p_mask)
    4. Uses prompt_lengths to separate prompt from response tokens

ReFusion is built on Qwen3 and uses AutoModelForCausalLM, so standard
HuggingFace/BnB/PEFT loading works without any compatibility patches.

Usage:
    uv run infra/training/flow_matching/flow_llm_sft_trainer.py
"""

import json
import logging
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
    FLOW_LLM_SFT_CONFIG,
)
from infra.training.shared.utils import format_chat_prompt, persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class FlowLLMDataset(Dataset):
    """Dataset for ReFusion masked diffusion training.

    Each item returns a full sequence (prompt + response) with:
    - input_ids: full token sequence
    - attention_mask: 1 for real tokens, 0 for padding
    - labels: -100 for prompt/padding, token IDs for response tokens
    - prompt_lengths: length of prompt (for ReFusion's forward_process)
    """

    def __init__(self, file_path: str, tokenizer, max_length: int, max_samples: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        if max_samples > 0:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} flow LLM training examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", item.get("condition", ""))
        target_text = item.get("response", "")
        if not target_text:
            target_actions = item.get("target", item.get("output", []))
            if isinstance(target_actions, list):
                target_text = "\n".join(target_actions)
            else:
                target_text = str(target_actions)

        # Tokenize prompt and response separately to track prompt_length
        prompt_enc = self.tokenizer(
            instruction, add_special_tokens=True, return_tensors="pt"
        )
        response_enc = self.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt"
        )

        prompt_ids = prompt_enc["input_ids"].squeeze(0)
        response_ids = response_enc["input_ids"].squeeze(0)
        prompt_length = prompt_ids.shape[0]

        # Concatenate and truncate/pad to max_length
        full_ids = torch.cat([prompt_ids, response_ids])[:self.max_length]
        full_length = full_ids.shape[0]

        # Labels: -100 for prompt tokens, token IDs for response tokens
        labels = torch.full((full_length,), IGNORE_INDEX, dtype=torch.long)
        response_start = min(prompt_length, full_length)
        labels[response_start:] = full_ids[response_start:]

        # Pad to max_length
        pad_length = self.max_length - full_length
        if pad_length > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            full_ids = torch.cat([full_ids, torch.full((pad_length,), pad_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_length,), IGNORE_INDEX, dtype=torch.long)])

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:full_length] = 1

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": torch.tensor([prompt_length], dtype=torch.long),
        }


def load_model_with_qlora(model_config: dict):
    """Load ReFusion with 4-bit quantization and LoRA adapters."""
    model_name = model_config["model_name"]
    compute_dtype = (
        torch.bfloat16
        if model_config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    trust_remote_code = model_config.get("trust_remote_code", True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config["load_in_4bit"],
        bnb_4bit_quant_type=model_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=model_config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ReFusion uses standard AutoModelForCausalLM -- no compatibility patches needed
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Apply LoRA (standard CAUSAL_LM task type)
    lora_config = LoraConfig(
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"],
        target_modules=model_config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train():
    """Run flow LLM SFT training."""
    model_config = FLOW_LLM_CONFIG
    train_config = FLOW_LLM_SFT_CONFIG

    # Load tokenizer
    trust_remote_code = model_config.get("trust_remote_code", True)
    logger.info(f"Loading tokenizer: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with QLoRA
    logger.info(f"Loading ReFusion with QLoRA: {model_config['model_name']}")
    model = load_model_with_qlora(model_config)

    # Load dataset
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    max_samples = DATA_CONFIG.get("max_train_samples", 0)
    dataset = FlowLLMDataset(
        train_file, tokenizer, model_config["max_seq_length"], max_samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # Optimizer (only LoRA params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    logger.info(
        f"Starting flow LLM SFT: {len(dataset)} samples, "
        f"{train_config['num_epochs']} epochs, "
        f"batch_size={train_config['batch_size']}, "
        f"grad_accum={train_config['gradient_accumulation_steps']}"
    )

    global_step = 0
    accumulation_steps = train_config["gradient_accumulation_steps"]
    device = next(model.parameters()).device

    for epoch in range(train_config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            # ReFusion forward() handles masked diffusion loss natively
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_lengths=prompt_lengths,
            )
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            batch_loss_val = loss.item() * accumulation_steps
            if not math.isnan(batch_loss_val):
                epoch_loss += batch_loss_val
                num_batches += 1
            else:
                logger.warning(f"NaN loss at batch {batch_idx}, skipping accumulation")

            if (batch_idx + 1) % accumulation_steps == 0:
                if global_step > 0 and global_step % train_config["logging_steps"] == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    logger.info(
                        f"Epoch {epoch + 1}, Step {global_step}: "
                        f"loss={batch_loss_val:.4f}, "
                        f"avg_loss={avg_loss:.4f}"
                    )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")

    # Save
    output_dir = Path("outputs/flow_llm_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    persist_checkpoint(str(output_dir), "flow-llm-sft")
    logger.info(f"Flow LLM SFT complete. Adapter saved to {output_dir / 'adapter'}")


if __name__ == "__main__":
    train()
