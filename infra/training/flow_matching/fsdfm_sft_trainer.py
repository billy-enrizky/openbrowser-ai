"""FS-DFM SFT trainer: Fine-tune FS-DFM 1.3B with LoRA on FormFactory data.

Loads pre-trained DFM_checkpoint.pth from HuggingFace, injects LoRA into
attention layers, and trains with generalized KL loss (discrete flow matching).

Loss is masked to response tokens only: instruction tokens (prefix) are frozen
during both training and generation via prefix conditioning / inpainting.

Usage:
    uv run infra/training/flow_matching/fsdfm_sft_trainer.py
"""

import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
    FSDFM_SFT_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    FSDFMTransformer,
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
    compute_generalized_kl_loss,
    inject_lora,
    load_fsdfm_from_huggingface,
    save_lora_weights,
)
from infra.training.shared.utils import persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FSDFMDataset(Dataset):
    """Dataset for FS-DFM SFT training.

    Each item returns:
        - input_ids: [max_length] full sequence (instruction + response), padded
        - loss_mask: [max_length] 1 for response tokens, 0 for instruction/padding
        - attention_mask: [max_length] 1 for real tokens, 0 for padding
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 1024,
        max_samples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        if max_samples > 0:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} FS-DFM SFT examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", item.get("condition", ""))
        response = item.get("response", "")
        if not response:
            target_actions = item.get("target", item.get("output", []))
            if isinstance(target_actions, list):
                response = "\n".join(target_actions)
            else:
                response = str(target_actions)

        # Tokenize instruction and response separately to find boundary
        inst_enc = self.tokenizer(
            instruction, add_special_tokens=True, return_tensors="pt"
        )
        resp_enc = self.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        )

        inst_ids = inst_enc["input_ids"].squeeze(0)
        resp_ids = resp_enc["input_ids"].squeeze(0)
        inst_len = inst_ids.shape[0]

        # Concatenate and truncate
        full_ids = torch.cat([inst_ids, resp_ids])[: self.max_length]
        full_len = full_ids.shape[0]

        # Loss mask: 0 for instruction, 1 for response tokens
        loss_mask = torch.zeros(full_len, dtype=torch.float32)
        resp_start = min(inst_len, full_len)
        loss_mask[resp_start:] = 1.0

        # Attention mask
        attention_mask = torch.ones(full_len, dtype=torch.long)

        # Pad to max_length
        pad_len = self.max_length - full_len
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            full_ids = torch.cat([
                full_ids,
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])
            loss_mask = torch.cat([
                loss_mask,
                torch.zeros(pad_len, dtype=torch.float32),
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.long),
            ])

        return {
            "input_ids": full_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }


def compute_fsdfm_loss(
    model: FSDFMTransformer,
    batch: dict,
    scheduler: PolynomialConvexScheduler,
    prob_path: MixtureDiscreteProbPath,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute flow matching loss for a batch.

    1. x_1 = clean tokens (input_ids)
    2. x_0 ~ Uniform(vocab_size) noise
    3. t ~ Uniform(eps, 1-eps)
    4. x_t = MixtureDiscreteProbPath.sample(x_0, x_1, t)
    5. logits = model(x_t, t)
    6. loss = generalized_kl_loss(logits, x_1, x_t, t, scheduler, loss_mask)
    """
    x_1 = batch["input_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    B, L = x_1.shape

    # Sample noise: x_0 ~ Uniform(vocab_size)
    x_0 = torch.randint(0, vocab_size, (B, L), device=device)

    # Sample timestep: t ~ Uniform(eps, 1-eps) for numerical stability
    eps = 1e-4
    t = torch.rand(B, device=device) * (1.0 - 2 * eps) + eps

    # Sample x_t from mixture path
    path_sample = prob_path.sample(x_0, x_1, t)
    x_t = path_sample.x_t

    # Forward pass
    logits = model(x_t, t)  # [B, L, V]

    # Compute generalized KL loss (masked to response tokens)
    loss = compute_generalized_kl_loss(
        logits=logits,
        x_1=x_1,
        x_t=x_t,
        t=t,
        scheduler=scheduler,
        loss_mask=loss_mask,
    )

    return loss


def train():
    """Run FS-DFM SFT training with LoRA."""
    model_config = FSDFM_MODEL_CONFIG
    train_config = FSDFM_SFT_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if train_config.get("bf16") else torch.float16

    # Load GPT-2 tokenizer (native to FS-DFM)
    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained FS-DFM from HuggingFace
    logger.info("Loading pre-trained FS-DFM 1.3B from HuggingFace")
    model = load_fsdfm_from_huggingface(model_config, device=device, dtype=compute_dtype)

    # Inject LoRA adapters
    logger.info("Injecting LoRA adapters")
    model = inject_lora(model, model_config)

    # Flow matching components
    exponent = model_config.get("scheduler_exponent", 2.0)
    scheduler = PolynomialConvexScheduler(exponent=exponent)
    prob_path = MixtureDiscreteProbPath(scheduler)

    # Load dataset
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    max_samples = DATA_CONFIG.get("max_train_samples", 0)
    dataset = FSDFMDataset(
        train_file,
        tokenizer,
        max_length=model_config["max_seq_length"],
        max_samples=max_samples,
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

    accumulation_steps = train_config["gradient_accumulation_steps"]
    grad_clip = train_config.get("grad_clip", 1.0)

    logger.info(
        f"Starting FS-DFM SFT: {len(dataset)} samples, "
        f"{train_config['num_epochs']} epochs, "
        f"batch_size={train_config['batch_size']}, "
        f"grad_accum={accumulation_steps}"
    )

    global_step = 0
    for epoch in range(train_config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            loss = compute_fsdfm_loss(
                model, batch, scheduler, prob_path, vocab_size, device
            )
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            batch_loss_val = loss.item() * accumulation_steps
            if not math.isnan(batch_loss_val):
                epoch_loss += batch_loss_val
                num_batches += 1
            else:
                logger.warning(f"NaN loss at batch {batch_idx}, skipping")

            if (batch_idx + 1) % accumulation_steps == 0:
                if global_step > 0 and global_step % train_config["logging_steps"] == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    logger.info(
                        f"Epoch {epoch + 1}, Step {global_step}: "
                        f"loss={batch_loss_val:.4f}, avg_loss={avg_loss:.4f}"
                    )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")

    # Save LoRA adapter weights
    output_dir = Path("outputs/fsdfm_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_lora_weights(model, str(adapter_path / "lora_weights.pt"))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"FS-DFM SFT complete. LoRA adapter saved to {adapter_path}")

    # Persist to Anyscale storage
    persist_checkpoint(str(output_dir), "fsdfm-sft")


if __name__ == "__main__":
    train()
