"""SFT trainer: LoRA fine-tuning on Mind2Web data."""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from infra.training.finetuning.config import SFT_CONFIG, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def format_prompt(example: dict) -> str:
    """Format instruction-response pair as chat prompt."""
    return (
        f"<|im_start|>system\n"
        f"You are a web browser automation agent. Given a task, "
        f"produce a step-by-step action plan to complete it.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}\n<|im_end|>"
    )


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize dataset for training."""
    def tokenize_fn(examples):
        prompts = [format_prompt({"instruction": inst, "response": resp})
                   for inst, resp in zip(examples["instruction"], examples["response"])]
        return tokenizer(
            prompts, truncation=True, max_length=max_length, padding="max_length"
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def train():
    """Run SFT training."""
    config = SFT_CONFIG

    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if config["fp16"] else torch.float32,
        device_map="auto",
    )

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
    dataset = load_sft_data(
        DATA_CONFIG["train_file"],
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Split train/eval
    split = dataset.train_test_split(test_size=DATA_CONFIG.get("eval_split", 0.1))
    train_dataset = tokenize_dataset(split["train"], tokenizer, config["max_seq_length"])
    eval_dataset = tokenize_dataset(split["test"], tokenizer, config["max_seq_length"])

    # Training args
    training_args = TrainingArguments(
        output_dir="outputs/finetuning_sft",
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting SFT training")
    trainer.train()
    trainer.save_model("outputs/finetuning_sft/final")
    tokenizer.save_pretrained("outputs/finetuning_sft/final")
    logger.info("SFT training complete")


if __name__ == "__main__":
    train()
