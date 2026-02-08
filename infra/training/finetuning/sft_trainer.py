"""SFT trainer: QLoRA fine-tuning on FormFactory data with Qwen3-8B.

Loads a pre-quantized 4-bit model, applies LoRA adapters, and trains
on instruction-response pairs with proper label masking (only response
tokens contribute to the loss).

Usage:
    uv run infra/training/finetuning/sft_trainer.py
"""

import json
import logging

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from infra.training.finetuning.config import DATA_CONFIG, S3_CONFIG, SFT_CONFIG
from infra.training.shared.utils import (
    format_prompt_parts,
    resolve_data_path,
    upload_checkpoint_to_s3,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def load_sft_data(file_path: str, max_samples: int = 0) -> Dataset:
    """Load SFT training data from JSONL."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples > 0:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} SFT examples")
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize dataset with proper label masking.

    Instruction tokens get label=-100 (ignored in loss).
    Only response tokens contribute to the training loss.
    """

    def tokenize_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for inst, resp in zip(examples["instruction"], examples["response"]):
            instruction_part, response_part = format_prompt_parts(inst, resp)

            # Tokenize instruction and response separately to find the boundary
            inst_tokens = tokenizer(
                instruction_part, add_special_tokens=False
            )
            resp_tokens = tokenizer(
                response_part, add_special_tokens=False
            )

            input_ids = inst_tokens["input_ids"] + resp_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)

            # Labels: mask instruction tokens with IGNORE_INDEX
            labels = (
                [IGNORE_INDEX] * len(inst_tokens["input_ids"])
                + resp_tokens["input_ids"]
            )

            # Truncate to max_length
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

            # Pad to max_length
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [IGNORE_INDEX] * pad_length

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    return dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )


def train():
    """Run SFT training with QLoRA."""
    config = SFT_CONFIG

    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    train_file = resolve_data_path(DATA_CONFIG["train_file"])

    dataset = load_sft_data(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Split train/eval
    split = dataset.train_test_split(
        test_size=DATA_CONFIG.get("eval_split", 0.1)
    )
    train_dataset = tokenize_dataset(
        split["train"], tokenizer, config["max_seq_length"]
    )
    eval_dataset = tokenize_dataset(
        split["test"], tokenizer, config["max_seq_length"]
    )

    # Training args
    output_dir = "outputs/finetuning_sft"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting SFT training (QLoRA on Qwen3-8B)")
    trainer.train()

    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"SFT training complete. Model saved to {final_dir}")

    # Upload checkpoint to S3
    upload_checkpoint_to_s3(final_dir, S3_CONFIG, "sft")


if __name__ == "__main__":
    train()
