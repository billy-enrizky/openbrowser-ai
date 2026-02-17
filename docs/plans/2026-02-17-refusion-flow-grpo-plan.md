# ReFusion Flow-GRPO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt Flow-GRPO (Liu et al., 2025) to ReFusion 8B masked diffusion model, aligning policy gradient optimization with the iterative unmasking generation process.

**Architecture:** Record the masked diffusion trajectory during generation (which positions were unmasked at each step and what tokens were chosen). During training, recompute per-step log-probabilities at the newly-unmasked positions with gradient flow, then apply PPO-style clipped surrogate objective with KL penalty against a frozen reference model.

**Tech Stack:** PyTorch, HuggingFace Transformers (4.52.4), PEFT/QLoRA, BitsAndBytes, Playwright (browser execution), Anyscale Ray

---

### Task 1: Add trajectory data structures to flow_llm_model.py

**Files:**
- Modify: `infra/training/flow_matching/flow_llm_model.py` (after line 34, before class FlowLLM)

**Step 1: Add dataclass imports and trajectory types**

Add at the top of the file after `import logging`:

```python
from dataclasses import dataclass, field
```

Add after `DEFAULT_MASK_TOKEN_ID = 151670` (line 36) and before `class FlowLLM`:

```python
@dataclass
class UnmaskingTrajectoryStep:
    """One denoising step in the masked diffusion trajectory.

    Records the masked state before this step and which positions were
    unmasked (with what tokens) so that per-step log-probs can be
    recomputed during training with gradient flow.
    """
    step_index: int
    masked_state: torch.Tensor         # [B, L_c + L_r] token IDs with mask tokens
    attention_mask: torch.Tensor       # [B, L_c + L_r]
    newly_unmasked_indices: list[list[int]]  # [B][k] indices into RESPONSE portion
    unmasked_tokens: list[list[int]]   # [B][k] token IDs placed at those positions


@dataclass
class UnmaskingTrajectory:
    """Full iterative unmasking trajectory for Flow-GRPO policy gradients."""
    steps: list[UnmaskingTrajectoryStep] = field(default_factory=list)
    final_tokens: torch.Tensor | None = None   # [B, L_r] fully unmasked response
    condition_length: int = 0                   # L_c (prompt length)
```

**Step 2: Commit**

```bash
git add infra/training/flow_matching/flow_llm_model.py
git commit -m "feat(training): add UnmaskingTrajectory data structures for ReFusion Flow-GRPO"
```

---

### Task 2: Add generate_with_trajectory() to FlowLLM

**Files:**
- Modify: `infra/training/flow_matching/flow_llm_model.py` (add method to FlowLLM class, after `generate()`)

**Step 1: Add trajectory-recording generation method**

Add this method to the `FlowLLM` class after the existing `generate()` method (after line 191):

```python
    @torch.no_grad()
    def generate_with_trajectory(
        self,
        condition_ids: torch.Tensor,
        condition_mask: torch.Tensor,
        seq_length: int,
        num_steps: int = 10,
        temperature: float = 0.7,
    ) -> UnmaskingTrajectory:
        """Generate via iterative unmasking, recording the trajectory.

        Same logic as generate() but stores (masked_state, unmasked_positions,
        unmasked_tokens) at each denoising step for Flow-GRPO training.

        Args:
            condition_ids: [B, L_c] prompt token IDs.
            condition_mask: [B, L_c] attention mask for prompt.
            seq_length: Number of response tokens to generate.
            num_steps: Number of denoising steps (T).
            temperature: Gumbel noise temperature (0 = greedy).

        Returns:
            UnmaskingTrajectory with T steps and final tokens.
        """
        self.model.eval()
        B = condition_ids.shape[0]
        L_c = condition_ids.shape[1]
        device = condition_ids.device

        trajectory = UnmaskingTrajectory(condition_length=L_c)

        # Start with fully masked response tokens
        current = torch.full(
            (B, seq_length), self.mask_token_id, dtype=torch.long, device=device
        )
        is_unmasked = torch.zeros(B, seq_length, dtype=torch.bool, device=device)

        for step in range(num_steps):
            t_next = (step + 1) / num_steps
            t_current = step / num_steps
            num_to_unmask = int(t_next * seq_length) - int(t_current * seq_length)

            if num_to_unmask <= 0 and step < num_steps - 1:
                continue

            # Build full input
            input_ids = torch.cat([condition_ids, current], dim=1)
            attn_mask = torch.cat(
                [condition_mask, torch.ones(B, seq_length, device=device, dtype=torch.long)],
                dim=1,
            )

            # Record the masked state BEFORE this step's unmasking
            step_record = UnmaskingTrajectoryStep(
                step_index=step,
                masked_state=input_ids.clone(),
                attention_mask=attn_mask.clone(),
                newly_unmasked_indices=[[] for _ in range(B)],
                unmasked_tokens=[[] for _ in range(B)],
            )

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, L_c:, :]  # [B, seq_length, V]

            # Gumbel noise for sampling diversity
            if temperature > 0:
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-10)
                )).float()
                perturbed_logits = logits / temperature + gumbel_noise
            else:
                perturbed_logits = logits

            predicted = perturbed_logits.argmax(dim=-1)  # [B, seq_length]

            # Confidence
            probs = F.softmax(logits, dim=-1)
            confidences = probs.gather(2, predicted.unsqueeze(-1)).squeeze(-1)
            confidences[is_unmasked] = -float("inf")

            if num_to_unmask > 0:
                remaining_masked = (~is_unmasked).sum(dim=-1).min().item()
                k = min(num_to_unmask, int(remaining_masked))
                if k > 0:
                    _, top_indices = confidences.topk(k, dim=-1)
                    for b in range(B):
                        for idx in top_indices[b]:
                            idx_val = idx.item()
                            current[b, idx_val] = predicted[b, idx_val]
                            is_unmasked[b, idx_val] = True
                            step_record.newly_unmasked_indices[b].append(idx_val)
                            step_record.unmasked_tokens[b].append(predicted[b, idx_val].item())

            # Only record steps that actually unmasked something
            if any(len(indices) > 0 for indices in step_record.newly_unmasked_indices):
                trajectory.steps.append(step_record)

        # Final pass: fill remaining masked positions
        if not is_unmasked.all():
            input_ids = torch.cat([condition_ids, current], dim=1)
            attn_mask = torch.cat(
                [condition_mask, torch.ones(B, seq_length, device=device, dtype=torch.long)],
                dim=1,
            )

            # Record this final step too
            final_step = UnmaskingTrajectoryStep(
                step_index=num_steps,
                masked_state=input_ids.clone(),
                attention_mask=attn_mask.clone(),
                newly_unmasked_indices=[[] for _ in range(B)],
                unmasked_tokens=[[] for _ in range(B)],
            )

            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, L_c:, :]
            predicted = logits.argmax(dim=-1)

            for b in range(B):
                for pos in range(seq_length):
                    if not is_unmasked[b, pos]:
                        current[b, pos] = predicted[b, pos]
                        final_step.newly_unmasked_indices[b].append(pos)
                        final_step.unmasked_tokens[b].append(predicted[b, pos].item())

            if any(len(indices) > 0 for indices in final_step.newly_unmasked_indices):
                trajectory.steps.append(final_step)

        trajectory.final_tokens = current
        return trajectory
```

**Step 2: Commit**

```bash
git add infra/training/flow_matching/flow_llm_model.py
git commit -m "feat(training): add generate_with_trajectory() to FlowLLM for ReFusion Flow-GRPO"
```

---

### Task 3: Add compute_unmasking_step_log_prob() function

**Files:**
- Modify: `infra/training/flow_matching/flow_llm_model.py` (add standalone function after class)

**Step 1: Add per-step log-prob computation**

Add after the FlowLLM class definition (at end of file):

```python
def compute_unmasking_step_log_prob(
    model,
    step: UnmaskingTrajectoryStep,
    condition_length: int,
) -> torch.Tensor:
    """Compute log-probability for one unmasking step (with gradient flow).

    Forward-passes the model with the recorded masked state, extracts
    response logits, and sums log_softmax at the positions that were
    newly unmasked in this step.

    Args:
        model: ReFusion policy or reference model (PEFT/QLoRA).
        step: Recorded trajectory step with masked_state and unmasked info.
        condition_length: L_c (prompt token count).

    Returns:
        [B] tensor of per-sample log-probabilities for this step.
    """
    B = step.masked_state.shape[0]
    device = step.masked_state.device

    # Forward pass (no labels, no prompt_lengths -- just logits)
    outputs = model(
        input_ids=step.masked_state,
        attention_mask=step.attention_mask,
    )
    # Response logits only
    response_logits = outputs.logits[:, condition_length:, :]  # [B, L_r, V]
    log_probs = F.log_softmax(response_logits, dim=-1)  # [B, L_r, V]

    # Sum log-probs at newly-unmasked positions
    step_log_prob = torch.zeros(B, device=device)
    for b in range(B):
        indices = step.newly_unmasked_indices[b]
        tokens = step.unmasked_tokens[b]
        if not indices:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
        step_log_prob[b] = log_probs[b, idx_t, tok_t].sum()

    return step_log_prob
```

**Step 2: Commit**

```bash
git add infra/training/flow_matching/flow_llm_model.py
git commit -m "feat(training): add compute_unmasking_step_log_prob() for ReFusion trajectory log-probs"
```

---

### Task 4: Add FLOW_GRPO_REFUSION_CONFIG to config.py

**Files:**
- Modify: `infra/training/flow_matching/config.py` (add after FLOW_GRPO_FSDFM_CONFIG block, before DATA_CONFIG)

**Step 1: Add configuration**

Add after the `FLOW_GRPO_FSDFM_CONFIG` block (after line 221) and before `DATA_CONFIG`:

```python
# --- Flow-GRPO for ReFusion 8B (masked diffusion policy gradients) ---
# Adapts Flow-GRPO to ReFusion's iterative unmasking process.  Per-step
# log-probabilities are computed at newly-unmasked positions, enabling
# PPO-style clipped policy gradients aligned with the generation trajectory.
# Fixes the generation/optimization mismatch in the existing GRPO trainer
# which used autoregressive log-probs despite masked diffusion generation.
FLOW_GRPO_REFUSION_CONFIG = {
    "group_size": 2,
    "learning_rate": 1e-5,
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "kl_coeff": 0.04,
    "clip_range": 0.2,
    "adv_clip_max": 5.0,
    "bf16": True,
    "logging_steps": 5,
    "grad_clip": 1.0,
    "num_generation_steps": 10,
    "generation_temperature": 0.7,
    "formfactory_port": int(os.environ.get("FORMFACTORY_PORT", "5050")),
    "browser_headless": True,
    "action_timeout_s": 5,
    "rollout_timeout_s": 30,
    "reward_weights": {
        "task_completion": 0.4,
        "field_accuracy": 0.4,
        "execution_completeness": 0.2,
    },
}
```

**Step 2: Commit**

```bash
git add infra/training/flow_matching/config.py
git commit -m "feat(training): add FLOW_GRPO_REFUSION_CONFIG for ReFusion 8B Flow-GRPO"
```

---

### Task 5: Create refusion_flow_grpo_trainer.py

**Files:**
- Create: `infra/training/flow_matching/refusion_flow_grpo_trainer.py`

**Reference files to follow:**
- `infra/training/flow_matching/fsdfm_flow_grpo_trainer.py` (structure/phases)
- `infra/training/flow_matching/online_flow_llm_grpo_trainer.py` (model loading pattern)

**Step 1: Write the trainer**

Create `infra/training/flow_matching/refusion_flow_grpo_trainer.py` with these sections:

1. **Module docstring**: Explain masked diffusion Flow-GRPO adaptation
2. **Imports**: torch, peft, transformers, FlowLLM, config, shared modules
3. **`load_quantized_model()`**: Copy from `online_flow_llm_grpo_trainer.py:65-89`
4. **`load_prompts()`**: Copy from `online_flow_llm_grpo_trainer.py:92-101`
5. **`async def train()`**: Main training function with 4 phases:
   - Model loading (policy + reference, both QLoRA, SFT checkpoint via PEFT)
   - Phase 1: Generate G rollouts with `flow_policy.generate_with_trajectory()`
   - Phase 2: Execute rollouts in browser, collect rewards
   - Phase 3: Compute group-relative advantages
   - Phase 4: PPO-style policy gradient over trajectory steps using `compute_unmasking_step_log_prob()`
6. **`if __name__ == "__main__": asyncio.run(train())`**

Key differences from `fsdfm_flow_grpo_trainer.py`:
- Uses `load_quantized_model()` + `PeftModel.from_pretrained()` instead of custom DiT loader
- Uses `FlowLLM` wrapper for generation
- Uses `compute_unmasking_step_log_prob()` instead of `compute_discrete_step_log_prob()`
- No scheduler (ReFusion uses linear unmasking, not polynomial flow)
- Imports `FLOW_GRPO_REFUSION_CONFIG` and `FLOW_LLM_CONFIG`

Key differences from `online_flow_llm_grpo_trainer.py`:
- Uses `generate_with_trajectory()` instead of `generate()`
- Iterates over trajectory steps instead of computing per-token AR log-probs
- PPO clipping instead of plain REINFORCE
- Advantage clipping (`adv_clip_max`)

The complete trainer code (see design doc for architecture details):

```python
"""ReFusion Flow-GRPO trainer: Masked diffusion policy gradients for ReFusion 8B.

Implements Flow-GRPO (Liu et al., 2025) adapted for ReFusion's iterative
unmasking process.  Unlike the existing online_flow_llm_grpo_trainer.py
which uses autoregressive log-probs, this computes per-step log-probabilities
at the positions that were actually unmasked during generation, aligning the
policy gradient with the masked diffusion trajectory.

Key innovation -- masked diffusion analog of continuous Flow-GRPO:
    Continuous: SDE step gives Gaussian policy N(mu, sigma^2 dt I)
                log-prob = Gaussian log-density
    ReFusion:   Each unmasking step predicts tokens for masked positions,
                unmasks top-k most confident.
                log-prob = sum of log_softmax at newly-unmasked positions

Architecture:
    1. Generate G rollouts via iterative unmasking, recording trajectories
    2. Execute rollouts in headless browser against FormFactory
    3. Compute group-relative advantages from browser rewards
    4. For each rollout, iterate over ALL trajectory steps (denoising reduction):
       a. Recompute policy log-prob at newly-unmasked positions (with gradients)
       b. Recompute old/reference log-prob (detached)
       c. PPO-style clipped surrogate objective on per-step log-ratios
       d. KL penalty from Schulman k3 approximation
    5. Average loss over steps and rollouts, backprop, update LoRA params

Reference: github.com/yifan123/flow_grpo (continuous version for images)

Usage:
    uv run infra/training/flow_matching/refusion_flow_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_GRPO_REFUSION_CONFIG,
    FLOW_LLM_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import (
    FlowLLM,
    compute_unmasking_step_log_prob,
)
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_quantized_model(model_name: str, config: dict):
    """Load ReFusion with 4-bit quantization."""
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    trust_remote_code = config.get("trust_remote_code", True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for Flow-GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info("Loaded %d prompts for ReFusion Flow-GRPO", len(records))
    return records


async def train():
    """Run ReFusion Flow-GRPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    grpo_config = FLOW_GRPO_REFUSION_CONFIG

    # Load tokenizer
    trust_remote_code = model_config.get("trust_remote_code", True)
    logger.info("Loading tokenizer: %s", model_config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine if loading from SFT checkpoint
    sft_checkpoint = os.environ.get("FLOW_LLM_SFT_CHECKPOINT", "")
    is_peft_checkpoint = sft_checkpoint and Path(sft_checkpoint).exists()

    # ---------------------------------------------------------------
    # Load policy model with QLoRA
    # ---------------------------------------------------------------
    if is_peft_checkpoint:
        logger.info("Loading SFT checkpoint: %s", sft_checkpoint)
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
                "SFT checkpoint not found at %s, training from base model",
                sft_checkpoint,
            )
        logger.info("Loading base model: %s", model_config["model_name"])
        policy_model = load_quantized_model(model_config["model_name"], model_config)
        policy_model.config.use_cache = False
        policy_model = prepare_model_for_kbit_training(
            policy_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=model_config["lora_target_modules"],
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)

    policy_model.print_trainable_parameters()

    # Wrap in FlowLLM for generation
    mask_token_id = model_config.get("mask_token_id", 151670)
    flow_policy = FlowLLM(policy_model, tokenizer, mask_token_id=mask_token_id)

    # ---------------------------------------------------------------
    # Load reference model (frozen, for KL penalty)
    # ---------------------------------------------------------------
    logger.info("Loading reference model (frozen)")
    ref_base = load_quantized_model(model_config["model_name"], model_config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_base, sft_checkpoint)
    else:
        ref_model = ref_base
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------------
    # Training data and optimizer
    # ---------------------------------------------------------------
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    clip_range = grpo_config["clip_range"]
    adv_clip_max = grpo_config.get("adv_clip_max", 5.0)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = grpo_config.get("num_generation_steps", 10)
    gen_temperature = grpo_config.get("generation_temperature", 0.7)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)
    grad_clip = grpo_config.get("grad_clip", 1.0)

    # ---------------------------------------------------------------
    # Start FormFactory server and browser
    # ---------------------------------------------------------------
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = grpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    headless = grpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting ReFusion Flow-GRPO: %d prompts, G=%d, kl=%.3f, clip=%.2f, "
        "T_gen=%d, temp=%.1f",
        len(prompts),
        group_size,
        kl_coeff,
        clip_range,
        num_gen_steps,
        gen_temperature,
    )

    total_steps = 0
    try:
        for epoch in range(grpo_config["num_epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, grpo_config["num_epochs"])
            epoch_rewards = []
            epoch_kl = []
            epoch_clipfrac = []

            for i, prompt_data in enumerate(prompts):
                instruction = prompt_data.get(
                    "instruction", prompt_data.get("condition", "")
                )
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})

                if not instruction or not form_url:
                    logger.warning("Skipping prompt %d: missing instruction or url", i)
                    continue

                # Periodic browser restart to reset DOM indices
                if i > 0 and i % 10 == 0:
                    logger.info("Periodic browser restart (prompt %d)", i)
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)

                # Tokenize condition
                condition_enc = tokenizer(
                    instruction,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                ).to(flow_policy.device)

                prompt_len = condition_enc["attention_mask"].sum().item()
                gen_length = max(1, max_seq_length - prompt_len)

                # ==========================================================
                # Phase 1: Generate G rollouts with trajectory recording
                # ==========================================================
                trajectories = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    trajectory = flow_policy.generate_with_trajectory(
                        condition_ids=condition_enc["input_ids"],
                        condition_mask=condition_enc["attention_mask"],
                        seq_length=gen_length,
                        num_steps=num_gen_steps,
                        temperature=gen_temperature,
                    )
                    trajectories.append(trajectory)
                    text = tokenizer.decode(
                        trajectory.final_tokens[0], skip_special_tokens=True
                    )
                    rollout_texts.append(text)
                policy_model.train()

                # ==========================================================
                # Phase 2: Execute each rollout in browser and score
                # ==========================================================
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
                        logger.warning("Navigation failed for rollout %d: %s", g, e)
                        rewards.append(0.0)
                        continue

                    actions = parse_rollout_to_actions(rollout_text, element_map)
                    if not actions:
                        logger.debug("No valid actions parsed from rollout %d", g)
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

                # ==========================================================
                # Phase 3: Compute group-relative advantages
                # ==========================================================
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # ==========================================================
                # Phase 4: Policy gradient update over trajectory steps
                # ==========================================================
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=flow_policy.device, requires_grad=False)
                total_kl = torch.tensor(0.0, device=flow_policy.device)
                total_clipfrac = 0.0
                valid_terms = 0

                for g in range(group_size):
                    traj = trajectories[g]
                    adv_g = torch.clamp(
                        advantages_t[g], -adv_clip_max, adv_clip_max
                    )

                    if len(traj.steps) == 0:
                        continue

                    for step in traj.steps:
                        # Current policy log-prob (WITH gradients)
                        log_prob = compute_unmasking_step_log_prob(
                            model=policy_model,
                            step=step,
                            condition_length=traj.condition_length,
                        )  # [B]

                        # Old policy log-prob (detached)
                        with torch.no_grad():
                            old_log_prob = compute_unmasking_step_log_prob(
                                model=policy_model,
                                step=step,
                                condition_length=traj.condition_length,
                            )  # [B]

                        # PPO clipped surrogate loss
                        ratio = torch.exp(log_prob - old_log_prob)  # [B]
                        unclipped = -adv_g * ratio
                        clipped = -adv_g * torch.clamp(
                            ratio, 1.0 - clip_range, 1.0 + clip_range
                        )
                        policy_loss = torch.maximum(unclipped, clipped).mean()

                        # Track clip fraction
                        with torch.no_grad():
                            clip_frac = (
                                (torch.abs(ratio - 1.0) > clip_range).float().mean()
                            )
                            total_clipfrac += clip_frac.item()

                        # KL penalty (Schulman k3)
                        kl_loss = torch.tensor(0.0, device=flow_policy.device)
                        if kl_coeff > 0:
                            with torch.no_grad():
                                ref_log_prob = compute_unmasking_step_log_prob(
                                    model=ref_model,
                                    step=step,
                                    condition_length=traj.condition_length,
                                )  # [B]
                            log_r = ref_log_prob - log_prob
                            kl_loss = (torch.exp(log_r) - log_r - 1).mean()
                            total_kl = total_kl + kl_loss.detach()

                        step_loss = policy_loss + kl_coeff * kl_loss
                        total_loss = total_loss + step_loss
                        valid_terms += 1

                # Average loss and backprop
                if valid_terms > 0:
                    loss = total_loss / valid_terms
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        optimizer.step()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = (total_kl / max(valid_terms, 1)).item()
                avg_clipfrac = total_clipfrac / max(valid_terms, 1)
                epoch_kl.append(avg_kl)
                epoch_clipfrac.append(avg_clipfrac)

                if total_steps % grpo_config["logging_steps"] == 0:
                    loss_val = (total_loss / max(valid_terms, 1)).item()
                    logger.info(
                        "  Step %d (prompt %d/%d): avg_reward=%.3f, "
                        "loss=%.4f, kl=%.4f, clipfrac=%.3f",
                        total_steps,
                        i + 1,
                        len(prompts),
                        avg_reward,
                        loss_val,
                        avg_kl,
                        avg_clipfrac,
                    )

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    "Epoch %d complete: avg_reward=%.3f, "
                    "nonzero_rewards=%d/%d, avg_kl=%.4f, avg_clipfrac=%.3f",
                    epoch + 1,
                    epoch_avg,
                    nonzero,
                    len(epoch_rewards),
                    sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0,
                    sum(epoch_clipfrac) / len(epoch_clipfrac) if epoch_clipfrac else 0,
                )

        # Save final model
        final_dir = Path("outputs/refusion_flow_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("ReFusion Flow-GRPO complete. Model saved to %s", final_dir)

        persist_checkpoint(str(final_dir), "refusion-flow-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
```

**Step 2: Commit**

```bash
git add infra/training/flow_matching/refusion_flow_grpo_trainer.py
git commit -m "feat(training): implement ReFusion Flow-GRPO trainer with masked diffusion policy gradients"
```

---

### Task 6: Add Anyscale job configs and register in submit_job.py

**Files:**
- Create: `infra/training/anyscale/refusion_flow_grpo_job.yaml`
- Create: `infra/training/anyscale/eval_refusion_flow_grpo_job.yaml`
- Modify: `infra/training/anyscale/submit_job.py` (add two entries to JOB_CONFIGS dict)

**Step 1: Create training job config**

Create `infra/training/anyscale/refusion_flow_grpo_job.yaml`:

```yaml
name: openbrowser-refusion-flow-grpo
entrypoint: python -m infra.training.flow_matching.refusion_flow_grpo_trainer
containerfile: infra/training/anyscale/Containerfile.online-grpo
compute_config:
  cloud: "Anyscale Cloud"
  head_node:
    instance_type: g6e.xlarge
  worker_nodes: []
working_dir: .
excludes:
  - .git
  - .env
  - __pycache__
  - node_modules
  - frontend
  - .cursor
  - presentation
  - data/mind2web/
  - data/webarena/
  - results/
  - outputs/
  - openbrowser_agent_data/
  - data/formfactory/.git/
  - data/formfactory/img/
  - "*.pyc"
env_vars:
  FLOW_TRAIN_FILE: "data/processed/formfactory_sft.jsonl"
  MAX_TRAIN_SAMPLES: "25"
  NUM_EPOCHS: "1"
  FLOW_LLM_SFT_CHECKPOINT: "/mnt/user_storage/openbrowser/checkpoints/flow-llm-sft/adapter"
  # HF_TOKEN injected from .env at submission time by submit_job.py
max_retries: 1
timeout_s: 14400
tags:
  project: openbrowser
  task: refusion-flow-grpo
  model: refusion-8b-mdm
```

**Step 2: Create evaluation job config**

Create `infra/training/anyscale/eval_refusion_flow_grpo_job.yaml`:

```yaml
name: openbrowser-eval-refusion-flow-grpo
entrypoint: python -m infra.training.flow_matching.eval_refusion_sft
containerfile: infra/training/anyscale/Containerfile.online-grpo
compute_config:
  cloud: "Anyscale Cloud"
  head_node:
    instance_type: g6e.xlarge
  worker_nodes: []
working_dir: .
excludes:
  - .git
  - .env
  - __pycache__
  - node_modules
  - frontend
  - .cursor
  - presentation
  - data/mind2web/
  - data/webarena/
  - results/
  - outputs/
  - openbrowser_agent_data/
  - data/formfactory/.git/
  - data/formfactory/img/
  - "*.pyc"
env_vars:
  FLOW_TRAIN_FILE: "data/processed/formfactory_sft.jsonl"
  MAX_EVAL_SAMPLES: "25"
  FLOW_LLM_SFT_CHECKPOINT: "/mnt/user_storage/openbrowser/checkpoints/refusion-flow-grpo"
  # HF_TOKEN injected from .env at submission time by submit_job.py
max_retries: 1
timeout_s: 7200
tags:
  project: openbrowser
  task: eval-refusion-flow-grpo
  model: refusion-8b-mdm
```

**Step 3: Register in submit_job.py**

Add two entries to `JOB_CONFIGS` dict (after the `"eval-fsdfm-flow-grpo"` entry at line 57):

```python
    "refusion-flow-grpo": JOBS_DIR / "refusion_flow_grpo_job.yaml",
    "eval-refusion-flow-grpo": JOBS_DIR / "eval_refusion_flow_grpo_job.yaml",
```

Also update the docstring Usage section to include:
```
    uv run infra/training/anyscale/submit_job.py refusion-flow-grpo
    uv run infra/training/anyscale/submit_job.py eval-refusion-flow-grpo
```

**Step 4: Commit**

```bash
git add infra/training/anyscale/refusion_flow_grpo_job.yaml \
        infra/training/anyscale/eval_refusion_flow_grpo_job.yaml \
        infra/training/anyscale/submit_job.py
git commit -m "feat(infra): add Anyscale job configs for ReFusion Flow-GRPO training and evaluation"
```

---

### Task 7: Update changelog and todo.md

**Files:**
- Modify: `local_docs/CHANGELOG.md` (add new entry at top)
- Modify: `infra/training/todo.md` (update next steps)

**Step 1: Update changelog**

Add a new `[Unreleased]` entry at the top with current timestamp (run `date` first):

```markdown
## [Unreleased] - YYYY-MM-DD HH:MM:SS

### Added

- `infra/training/flow_matching/flow_llm_model.py`: Added `UnmaskingTrajectory`/`UnmaskingTrajectoryStep` dataclasses, `generate_with_trajectory()` method for trajectory-recording iterative unmasking, `compute_unmasking_step_log_prob()` for per-step masked log-probabilities
- `infra/training/flow_matching/refusion_flow_grpo_trainer.py`: Flow-GRPO trainer for ReFusion 8B -- adapts Flow-GRPO (Liu et al., 2025) to masked diffusion unmasking, computing per-step log-probs at newly-unmasked positions
- `infra/training/flow_matching/config.py`: Added `FLOW_GRPO_REFUSION_CONFIG` with PPO-style clipping, KL penalty, and denoising reduction (T=10)
- `infra/training/anyscale/refusion_flow_grpo_job.yaml`: Anyscale job config for ReFusion Flow-GRPO training
- `infra/training/anyscale/eval_refusion_flow_grpo_job.yaml`: Anyscale job config for ReFusion Flow-GRPO evaluation
- `infra/training/anyscale/submit_job.py`: Registered `refusion-flow-grpo` and `eval-refusion-flow-grpo` job names
- `docs/plans/2026-02-17-refusion-flow-grpo-design.md`: Design document for ReFusion Flow-GRPO adaptation
```

**Step 2: Update todo.md**

Add ReFusion Flow-GRPO items to the next steps:

```markdown
- [ ] Submit ReFusion Flow-GRPO job: `uv run infra/training/anyscale/submit_job.py refusion-flow-grpo`
- [ ] After training, submit eval: `uv run infra/training/anyscale/submit_job.py eval-refusion-flow-grpo`
- [ ] Compare ReFusion Flow-GRPO results with existing baselines
```

**Step 3: Commit**

```bash
git add infra/training/todo.md
git commit -m "docs: update todo with ReFusion Flow-GRPO implementation and next steps"
```

Note: `local_docs/CHANGELOG.md` is gitignored, so it won't be committed.
