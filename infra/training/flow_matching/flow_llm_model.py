"""Masked Diffusion with ReFusion backbone.

Uses GSAI-ML/ReFusion, a masked diffusion LLM built on Qwen3. ReFusion
supports both autoregressive and masked diffusion decoding natively:

Training (handled by ReFusion's forward() internally):
    1. Split response into slots of random size (4/8/16/32 tokens)
    2. Randomly mask some slots, keep others in shuffled order
    3. AR loss on unmasked slots + MDM loss on masked slots / p_mask
    4. forward() accepts prompt_lengths to separate prompt from response

Generation (iterative unmasking via FlowLLM wrapper):
    1. Start with prompt + fully masked response positions
    2. Predict all masked positions simultaneously
    3. Accept tokens above confidence threshold
    4. Repeat until all positions are unmasked or max steps reached

Architecture:
    - Backbone: ReFusion (Qwen3ForCausalLM, ~8B params)
    - Quantization: QLoRA (4-bit NF4 + LoRA adapters)
    - Mask token ID: 151670
    - Vocabulary: 151671 tokens

This is the STAD80 counterpart to the STAD68 autoregressive approach.
    - STAD68 (AR): Qwen3-8B, left-to-right token generation
    - STAD80 (MDM): ReFusion, slot-based parallel masked diffusion
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MASK_TOKEN_ID = 151670


class FlowLLM:
    """Wraps ReFusion for masked diffusion generation.

    For training, ReFusion's forward() handles the masked diffusion loss
    internally when passed (input_ids, attention_mask, labels, prompt_lengths).
    This wrapper is mainly needed for GRPO generation (iterative unmasking).
    """

    def __init__(self, model, tokenizer, mask_token_id: int = DEFAULT_MASK_TOKEN_ID):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.vocab_size = getattr(model.config, "vocab_size", 151671)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"FlowLLM (ReFusion): {total_params / 1e9:.1f}B total, "
            f"{trainable_params / 1e6:.1f}M trainable (QLoRA), "
            f"mask_token_id={mask_token_id}"
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def compute_sft_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ReFusion's native masked diffusion SFT loss.

        ReFusion's forward() handles the forward process internally:
        slots of random size are masked, AR loss on unmasked, MDM loss on masked.

        Args:
            input_ids: [B, L] full sequence (prompt + response + padding).
            attention_mask: [B, L] attention mask.
            labels: [B, L] labels (-100 for prompt/padding, token IDs for response).
            prompt_lengths: [B, 1] length of prompt for each sample.

        Returns:
            Scalar loss tensor.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt_lengths=prompt_lengths,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        condition_ids: torch.Tensor,
        condition_mask: torch.Tensor,
        seq_length: int,
        num_steps: int = 64,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """Generate via iterative unmasking (masked diffusion reverse process).

        Starting from fully masked response positions, progressively unmask
        tokens over num_steps iterations using confidence-based scheduling.

        Args:
            condition_ids: [B, L_c] condition token IDs (prompt).
            condition_mask: [B, L_c] attention mask for condition.
            seq_length: Number of response tokens to generate.
            num_steps: Number of denoising steps.
            temperature: Gumbel noise temperature (0 = greedy).

        Returns:
            [B, seq_length] generated token IDs.
        """
        self.model.eval()
        B = condition_ids.shape[0]
        L_c = condition_ids.shape[1]
        device = condition_ids.device

        # Start with fully masked response tokens
        current = torch.full(
            (B, seq_length), self.mask_token_id, dtype=torch.long, device=device
        )
        is_unmasked = torch.zeros(B, seq_length, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # Linear unmasking schedule
            t_next = (step + 1) / num_steps
            t_current = step / num_steps
            num_to_unmask = int(t_next * seq_length) - int(t_current * seq_length)

            if num_to_unmask <= 0 and step < num_steps - 1:
                continue

            # Build full input: prompt + current (partially masked) response
            input_ids = torch.cat([condition_ids, current], dim=1)
            attn_mask = torch.cat(
                [condition_mask, torch.ones(B, seq_length, device=device, dtype=torch.long)],
                dim=1,
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

            # Confidence: softmax probability of predicted token
            probs = F.softmax(logits, dim=-1)
            confidences = probs.gather(2, predicted.unsqueeze(-1)).squeeze(-1)

            # Only consider currently masked positions
            confidences[is_unmasked] = -float("inf")

            if num_to_unmask > 0:
                remaining_masked = (~is_unmasked).sum(dim=-1).min().item()
                k = min(num_to_unmask, int(remaining_masked))
                if k > 0:
                    _, top_indices = confidences.topk(k, dim=-1)
                    for b in range(B):
                        for idx in top_indices[b]:
                            current[b, idx] = predicted[b, idx]
                            is_unmasked[b, idx] = True

        # Final pass: fill any remaining masked positions
        if not is_unmasked.all():
            input_ids = torch.cat([condition_ids, current], dim=1)
            attn_mask = torch.cat(
                [condition_mask, torch.ones(B, seq_length, device=device, dtype=torch.long)],
                dim=1,
            )
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, L_c:, :]
            predicted = logits.argmax(dim=-1)
            current = torch.where(is_unmasked, current, predicted)

        return current
