"""Online Flow LLM GRPO trainer: LLaDA-8B masked diffusion with browser execution.

Uses LLaDA-8B (GSAI-ML/LLaDA-8B-Base) with QLoRA as the backbone for
masked diffusion, trained with GRPO (Group Relative Policy Optimization)
using real browser execution rewards from FormFactory.

Architecture:
    1. FlowLLM (LLaDA-8B + QLoRA) generates G candidate plans via
       iterative unmasking (masked diffusion reverse process)
    2. Each plan is decoded to text and parsed into executable actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = actual form submission success + field accuracy
    5. Advantage-weighted denoising loss updates LoRA parameters
    6. KL penalty against a frozen reference model for stability

This is the STAD80 counterpart to the AR GRPO trainer (STAD68).
    - STAD68: Qwen3-8B, left-to-right autoregressive token generation
    - STAD80: LLaDA-8B, parallel iterative unmasking via masked diffusion

Usage:
    uv run infra/training/flow_matching/online_flow_llm_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

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
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import (
    format_chat_prompt,
    persist_checkpoint,
    resolve_data_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _patch_bnb_for_llada():
    """Patch BitsAndBytes quantizer and model class for LLaDA compatibility.

    LLaDA's custom model class (LLaDAModelLM) has two incompatibilities with
    the version of transformers on the Anyscale cluster:
    1. `all_tied_weights_keys` is a list but transformers expects a dict (.keys())
    2. `tie_weights()` doesn't accept `missing_keys`/`recompute_mapping` kwargs
       that newer transformers passes during `_finalize_model_loading`
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
    # This triggers the custom code download and registers the model class
    from transformers import AutoModel
    model_cls = AutoModel._model_mapping[type(config)]

    import inspect
    orig_tie = model_cls.tie_weights
    sig = inspect.signature(orig_tie)
    if "missing_keys" not in sig.parameters:
        def _compat_tie_weights(self, **kwargs):
            return orig_tie(self)
        model_cls.tie_weights = _compat_tie_weights


def load_quantized_model(model_name: str, config: dict):
    """Load LLaDA-8B with 4-bit quantization."""
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    trust_remote_code = config.get("trust_remote_code", True)

    # Patch BnB for LLaDA compatibility (missing all_tied_weights_keys)
    _patch_bnb_for_llada()
    # Patch tie_weights for LLaDA (missing kwargs in newer transformers)
    _patch_llada_tie_weights(model_name, trust_remote_code)

    # LLaDA has no pre-quantized variant -- always quantize on-the-fly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
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
    return model


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for online flow LLM GRPO")
    return records


async def train():
    """Run online flow LLM GRPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    grpo_config = ONLINE_FLOW_GRPO_CONFIG

    # Load tokenizer
    trust_remote_code = model_config.get("trust_remote_code", True)
    logger.info(f"Loading tokenizer: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine if loading from SFT checkpoint
    sft_checkpoint = os.environ.get("FLOW_LLM_SFT_CHECKPOINT", "")
    is_peft_checkpoint = sft_checkpoint and Path(sft_checkpoint).exists()

    # Load policy model with QLoRA
    if is_peft_checkpoint:
        logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")
        base_model = load_quantized_model(model_config["model_name"], model_config)
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        policy_model = PeftModel.from_pretrained(
            base_model, sft_checkpoint, is_trainable=True
        )
        policy_model.train()
    else:
        if sft_checkpoint:
            logger.warning(
                f"SFT checkpoint not found at {sft_checkpoint}, "
                "training from base model"
            )
        logger.info(f"Loading base model: {model_config['model_name']}")
        policy_model = load_quantized_model(model_config["model_name"], model_config)
        policy_model.config.use_cache = False
        policy_model = prepare_model_for_kbit_training(
            policy_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # LLaDA is not a causal LM -- use FEATURE_EXTRACTION task type
        # since it's a bidirectional encoder with custom head
        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=model_config["lora_target_modules"],
            task_type="FEATURE_EXTRACTION",
        )
        policy_model = get_peft_model(policy_model, lora_config)

    policy_model.print_trainable_parameters()

    # Wrap in FlowLLM (LLaDA masked diffusion)
    mask_token_id = model_config.get("mask_token_id", 126336)
    flow_policy = FlowLLM(policy_model, tokenizer, mask_token_id=mask_token_id)

    # Load reference model (frozen, for KL computation)
    logger.info("Loading reference model (frozen)")
    ref_base = load_quantized_model(model_config["model_name"], model_config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_base, sft_checkpoint)
    else:
        ref_model = ref_base
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    flow_ref = FlowLLM(ref_model, tokenizer, mask_token_id=mask_token_id)

    # Load training data
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Optimizer (only LoRA params)
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    clip_range = grpo_config["clip_range"]
    max_seq_length = model_config["max_seq_length"]
    num_denoising_steps = grpo_config.get(
        "num_denoising_steps", model_config.get("num_denoising_steps", 20)
    )
    gen_temperature = model_config.get("generation_temperature", 0.7)
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
        f"Starting online flow LLM GRPO: {len(prompts)} prompts, G={group_size}, "
        f"kl_coeff={kl_coeff}, clip_range={clip_range}, "
        f"denoising_steps={num_denoising_steps}"
    )

    total_steps = 0
    try:
        for epoch in range(grpo_config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{grpo_config['num_epochs']}")
            epoch_rewards = []
            epoch_kl = []

            for i, prompt_data in enumerate(prompts):
                instruction = prompt_data.get(
                    "instruction", prompt_data.get("condition", "")
                )
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})
                # Support multiple data formats: response (formfactory_sft), target, output
                target_text = prompt_data.get("response", "")
                if not target_text:
                    target_actions = prompt_data.get("target", prompt_data.get("output", []))
                    target_text = (
                        "\n".join(target_actions)
                        if isinstance(target_actions, list)
                        else str(target_actions)
                    )

                if not instruction or not form_url:
                    logger.warning(
                        f"Skipping prompt {i}: missing instruction or url"
                    )
                    continue

                # Tokenize condition
                condition_enc = tokenizer(
                    instruction,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(flow_policy.device)

                # Generate G rollouts via iterative denoising
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    generated_ids = flow_policy.generate(
                        condition_ids=condition_enc["input_ids"],
                        condition_mask=condition_enc["attention_mask"],
                        seq_length=max_seq_length,
                        num_steps=num_denoising_steps,
                        temperature=gen_temperature,
                    )
                    # Decode generated tokens to text
                    text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    rollout_texts.append(text)
                policy_model.train()

                # Execute each rollout in browser and score
                rewards = []
                for g, rollout_text in enumerate(rollout_texts):
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
                        logger.warning(f"Navigation failed for rollout {g}: {e}")
                        rewards.append(0.0)
                        continue

                    actions = parse_rollout_to_actions(rollout_text, element_map)
                    if not actions:
                        logger.debug(f"No valid actions parsed from rollout {g}")
                        rewards.append(0.0)
                        continue

                    outcome = await browser_env.execute_actions(
                        actions, timeout_per_action=action_timeout
                    )
                    reward = compute_online_reward(
                        outcome,
                        ground_truth_fields,
                        weights=grpo_config.get("reward_weights"),
                    )
                    rewards.append(reward)

                epoch_rewards.extend(rewards)

                # Compute GRPO advantages
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # Tokenize target for loss computation
                target_enc = tokenizer(
                    target_text,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(flow_policy.device)

                # Compute advantage-weighted denoising loss
                total_loss = torch.tensor(0.0, device=flow_policy.device)
                total_kl = torch.tensor(0.0, device=flow_policy.device)

                for g in range(group_size):
                    # Policy denoising loss
                    policy_loss = flow_policy.compute_loss(
                        condition_enc["input_ids"],
                        condition_enc["attention_mask"],
                        target_enc["input_ids"],
                        target_enc["attention_mask"],
                    )

                    # Reference denoising loss (for KL estimation)
                    with torch.no_grad():
                        ref_loss = flow_ref.compute_loss(
                            condition_enc["input_ids"],
                            condition_enc["attention_mask"],
                            target_enc["input_ids"],
                            target_enc["attention_mask"],
                        )

                    # KL divergence estimate: difference in denoising losses
                    # Lower policy loss = better denoising = diverged from ref
                    kl_estimate = ref_loss - policy_loss

                    # Advantage-weighted loss with clipping
                    ratio = torch.exp(-(policy_loss - ref_loss))
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    pg_loss1 = -advantages_t[g] * ratio * policy_loss
                    pg_loss2 = -advantages_t[g] * clipped_ratio * policy_loss
                    pg_loss = torch.max(pg_loss1, pg_loss2)

                    total_loss = total_loss + pg_loss + kl_coeff * kl_estimate
                    total_kl = total_kl + kl_estimate

                loss = total_loss / group_size

                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = (total_kl / group_size).item()
                epoch_kl.append(avg_kl)

                if total_steps % grpo_config["logging_steps"] == 0:
                    logger.info(
                        f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                        f"avg_reward={avg_reward:.3f}, "
                        f"loss={loss.item():.4f}, "
                        f"kl={avg_kl:.4f}"
                    )

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    f"Epoch {epoch + 1} complete: avg_reward={epoch_avg:.3f}, "
                    f"nonzero_rewards={nonzero}/{len(epoch_rewards)}, "
                    f"avg_kl={sum(epoch_kl) / len(epoch_kl):.4f}"
                )

        # Save final model
        final_dir = Path("outputs/flow_llm_online_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Flow LLM GRPO complete. Model saved to {final_dir}")

        persist_checkpoint(str(final_dir), "online-flow-llm-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
