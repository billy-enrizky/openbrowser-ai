"""Flow LLM SFT trainer: Masked diffusion denoising on LLaDA-8B.

Trains LLaDA-8B (GSAI-ML/LLaDA-8B-Base) with QLoRA for form-filling
action plan generation via masked diffusion. At each training step:
    1. Sample time t ~ Uniform(eps, 1-eps) per batch element
    2. Mask response tokens with probability p_mask = (1-eps)*t + eps
    3. Feed prompt (unmasked) + masked response to bidirectional model
    4. Compute cross-entropy loss on masked positions, normalized by p_mask

LLaDA natively supports masked diffusion -- it was pre-trained for this
exact objective. This SFT stage teaches it the form-filling action format.

Usage:
    uv run infra/training/flow_matching/flow_llm_sft_trainer.py
"""

import json
import logging
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
    FLOW_LLM_SFT_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import FlowLLM
from infra.training.shared.utils import format_chat_prompt, persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FlowLLMDataset(Dataset):
    """Dataset for flow LLM denoising training."""

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
        # Support multiple data formats: response (formfactory_sft), target, output
        target_text = item.get("response", "")
        if not target_text:
            target_actions = item.get("target", item.get("output", []))
            if isinstance(target_actions, list):
                target_text = "\n".join(target_actions)
            else:
                target_text = str(target_actions)

        # Tokenize condition (instruction)
        condition = self.tokenizer(
            instruction,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target (action plan)
        target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "condition_ids": condition["input_ids"].squeeze(0),
            "condition_mask": condition["attention_mask"].squeeze(0),
            "target_ids": target["input_ids"].squeeze(0),
            "target_mask": target["attention_mask"].squeeze(0),
        }


def _patch_bnb_for_llada():
    """Patch BitsAndBytes quantizer for LLaDA compatibility.

    LLaDA's custom model class sets `all_tied_weights_keys` as a list
    but transformers expects a dict (.keys()).
    """
    import transformers.quantizers.base as _qbase

    _orig_fn = _qbase.get_keys_to_not_convert

    def _patched_fn(model):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        elif isinstance(model.all_tied_weights_keys, list):
            model.all_tied_weights_keys = {k: None for k in model.all_tied_weights_keys}
        return _orig_fn(model)

    _qbase.get_keys_to_not_convert = _patched_fn


def _patch_llada_tie_weights(model_name: str, trust_remote_code: bool = True):
    """Patch LLaDA's tie_weights to accept kwargs from newer transformers.

    Newer transformers calls model.tie_weights(missing_keys=..., recompute_mapping=...)
    but LLaDA's custom tie_weights() doesn't accept those kwargs. We patch the
    model class BEFORE from_pretrained so the fix is in place during loading.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    from transformers import AutoModel
    model_cls = AutoModel._model_mapping[type(config)]

    import inspect
    orig_tie = model_cls.tie_weights
    sig = inspect.signature(orig_tie)
    if "missing_keys" not in sig.parameters:
        def _compat_tie_weights(self, **kwargs):
            return orig_tie(self)
        model_cls.tie_weights = _compat_tie_weights


def load_model_with_qlora(model_config: dict, lora_config_dict: dict):
    """Load LLaDA-8B with 4-bit quantization and LoRA adapters."""
    model_name = model_config["model_name"]
    compute_dtype = (
        torch.bfloat16
        if model_config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    trust_remote_code = model_config.get("trust_remote_code", True)

    # Patch BnB for LLaDA compatibility (missing all_tied_weights_keys)
    _patch_bnb_for_llada()
    # Patch tie_weights for LLaDA (missing kwargs in newer transformers)
    _patch_llada_tie_weights(model_name, trust_remote_code)

    # LLaDA has no pre-quantized variant -- always quantize on-the-fly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config["load_in_4bit"],
        bnb_4bit_quant_type=model_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=model_config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # LLaDA uses AutoModel (not AutoModelForCausalLM) + trust_remote_code
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_config_dict["lora_r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        target_modules=lora_config_dict["lora_target_modules"],
        task_type="FEATURE_EXTRACTION",
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
    logger.info(f"Loading LLaDA-8B with QLoRA: {model_config['model_name']}")
    model = load_model_with_qlora(model_config, model_config)

    # Wrap in FlowLLM (LLaDA masked diffusion)
    mask_token_id = model_config.get("mask_token_id", 126336)
    flow_model = FlowLLM(model, tokenizer, mask_token_id=mask_token_id)

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

    # Warmup scheduler
    total_steps = len(dataloader) * train_config["num_epochs"]
    warmup_steps = int(total_steps * train_config.get("warmup_ratio", 0.05))

    logger.info(
        f"Starting flow LLM SFT: {len(dataset)} samples, "
        f"{train_config['num_epochs']} epochs, "
        f"batch_size={train_config['batch_size']}, "
        f"grad_accum={train_config['gradient_accumulation_steps']}"
    )

    global_step = 0
    accumulation_steps = train_config["gradient_accumulation_steps"]

    for epoch in range(train_config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            condition_ids = batch["condition_ids"].to(flow_model.device)
            condition_mask = batch["condition_mask"].to(flow_model.device)
            target_ids = batch["target_ids"].to(flow_model.device)
            target_mask = batch["target_mask"].to(flow_model.device)

            # Flow matching denoising loss
            loss = flow_model.compute_loss(
                condition_ids, condition_mask, target_ids, target_mask
            )
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1

            if global_step > 0 and global_step % train_config["logging_steps"] == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Epoch {epoch + 1}, Step {global_step}: "
                    f"loss={loss.item() * accumulation_steps:.4f}, "
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
