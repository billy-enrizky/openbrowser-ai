"""Discrete Masked Diffusion with LLaDA-8B backbone.

Uses GSAI-ML/LLaDA-8B-Base, a pre-trained masked diffusion language model.
Unlike autoregressive LLMs, LLaDA uses bidirectional attention and generates
via iterative unmasking: tokens start fully masked (mask token ID 126336)
and are progressively revealed over multiple denoising steps.

Training (forward process):
    1. Sample time t ~ Uniform(eps, 1-eps) per batch element
    2. Mask each response token independently with probability (1-eps)*t + eps
    3. Feed prompt (unmasked) + masked response to bidirectional model
    4. Compute cross-entropy loss on masked positions, normalized by p_mask

Generation (reverse process / iterative unmasking):
    1. Start with prompt tokens + fully masked response positions
    2. At each step, predict all masked positions simultaneously
    3. Apply Gumbel noise for sampling diversity
    4. Unmask the most confident predictions; remask the rest
    5. Repeat until all positions are unmasked

Architecture:
    - Backbone: LLaDA-8B (bidirectional transformer encoder, ~8B params)
    - Quantization: QLoRA (4-bit NF4 + LoRA adapters)
    - Mask token ID: 126336
    - Vocabulary: 126464 tokens

This is the STAD80 counterpart to the STAD68 autoregressive approach.
    - STAD68 (AR): Qwen3-8B, left-to-right token generation
    - STAD80 (MDM): LLaDA-8B, parallel iterative unmasking
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# LLaDA mask token ID (from tokenizer: <|mdm_mask|>)
DEFAULT_MASK_TOKEN_ID = 126336


class FlowLLM:
    """Wraps LLaDA-8B for masked diffusion training and generation.

    The forward process masks response tokens at a variable rate,
    and the model learns to predict the original tokens at masked positions.
    At inference time, iterative unmasking from fully masked tokens produces
    the output.

    This class is a lightweight wrapper -- the actual model is a LLaDA
    model loaded via AutoModel with QLoRA adapters. The wrapper provides
    the masking, loss, and generation logic following the LLaDA paper.
    """

    def __init__(self, model, tokenizer, mask_token_id: int = DEFAULT_MASK_TOKEN_ID):
        """Initialize FlowLLM wrapper.

        Args:
            model: LLaDA model loaded via AutoModel (with or without PEFT adapters).
            tokenizer: HuggingFace tokenizer for the model.
            mask_token_id: The mask token ID used by LLaDA (default: 126336).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.vocab_size = getattr(model.config, "vocab_size", 126464)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"FlowLLM (LLaDA-8B): {total_params / 1e9:.1f}B total, "
            f"{trainable_params / 1e6:.1f}M trainable (QLoRA), "
            f"mask_token_id={mask_token_id}"
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward_process(
        self, input_ids: torch.Tensor, eps: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply LLaDA forward process: variable-rate token masking.

        Following the LLaDA paper, each batch element gets a random masking
        probability p_mask = (1 - eps) * t + eps, where t ~ Uniform(0, 1).
        Each token is independently masked with probability p_mask.

        Args:
            input_ids: [B, L] original token IDs (response tokens only).
            eps: Small constant to avoid p_mask=0 (no masking).

        Returns:
            noisy_ids: [B, L] token IDs with some positions replaced by mask token.
            masked_indices: [B, L] boolean mask (True = position was masked).
            p_mask: [B, L] masking probability at each position.
        """
        B, L = input_ids.shape

        # Sample random timestep per batch element
        t = torch.rand(B, device=input_ids.device)
        p_mask = (1 - eps) * t + eps  # [B]
        p_mask = p_mask[:, None].expand(B, L)  # [B, L]

        # Mask each token independently with probability p_mask
        masked_indices = torch.rand((B, L), device=input_ids.device) < p_mask

        # Replace masked positions with the mask token
        noisy_ids = torch.where(masked_indices, self.mask_token_id, input_ids)

        return noisy_ids, masked_indices, p_mask

    def compute_loss(
        self,
        condition_ids: torch.Tensor,
        condition_mask: torch.Tensor,
        clean_ids: torch.Tensor,
        clean_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LLaDA masked diffusion loss.

        Following the LLaDA paper: SFT does not mask the prompt tokens.
        Only response tokens are subject to masking. The loss is cross-entropy
        on masked positions, normalized by the masking probability p_mask.

        Args:
            condition_ids: [B, L_c] condition/instruction token IDs (never masked).
            condition_mask: [B, L_c] attention mask for condition.
            clean_ids: [B, L_t] target clean token IDs (response).
            clean_mask: [B, L_t] attention mask for target (1=real, 0=padding).

        Returns:
            Scalar loss tensor.
        """
        B = condition_ids.shape[0]

        # Apply forward process to response tokens only (prompt stays unmasked)
        noisy_ids, masked_indices, p_mask = self.forward_process(clean_ids)

        # Only compute loss on masked positions that are real tokens (not padding)
        valid_mask = masked_indices & clean_mask.bool()  # [B, L_t]

        # Concatenate prompt (unmasked) + masked response
        input_ids = torch.cat([condition_ids, noisy_ids], dim=1)
        attention_mask = torch.cat([condition_mask, clean_mask], dim=1)

        # Forward through LLaDA (bidirectional attention)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Extract logits for the response portion only
        logits = outputs.logits[:, condition_ids.shape[1]:, :]  # [B, L_t, V]

        if valid_mask.any():
            # Cross-entropy on masked positions, normalized by masking probability
            # Following LLaDA: loss = CE(logits[masked], labels[masked]) / p_mask[masked]
            token_loss = F.cross_entropy(
                logits[valid_mask],  # [num_masked, V]
                clean_ids[valid_mask],  # [num_masked]
                reduction="none",
            )
            # Normalize each token's loss by its masking probability
            normalized_loss = token_loss / p_mask[valid_mask]
            loss = normalized_loss.mean()
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return loss

    @torch.no_grad()
    def generate(
        self,
        condition_ids: torch.Tensor,
        condition_mask: torch.Tensor,
        seq_length: int,
        num_steps: int = 64,
        temperature: float = 0.7,
        remasking: str = "low_confidence",
    ) -> torch.Tensor:
        """Generate via iterative unmasking (LLaDA reverse process).

        Starting from fully masked response positions, progressively unmask
        tokens over num_steps iterations. At each step:
        1. Predict logits for all positions
        2. Add Gumbel noise for sampling diversity
        3. Compute confidence scores
        4. Unmask the top-k most confident positions (k determined by schedule)
        5. Remask the rest

        Args:
            condition_ids: [B, L_c] condition token IDs (prompt).
            condition_mask: [B, L_c] attention mask for condition.
            seq_length: Number of response tokens to generate.
            num_steps: Number of denoising steps (more = better quality).
            temperature: Sampling temperature for Gumbel noise.
            remasking: Strategy for remasking -- "low_confidence" (default) or "random".

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
        # Track which positions have been unmasked
        is_unmasked = torch.zeros(B, seq_length, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # Determine how many tokens to unmask at this step (linear schedule)
            t_next = (step + 1) / num_steps
            t_current = step / num_steps
            num_to_unmask = int(t_next * seq_length) - int(t_current * seq_length)

            if num_to_unmask <= 0 and step < num_steps - 1:
                continue

            # Build full input: prompt (unmasked) + current response (partially masked)
            input_ids = torch.cat([condition_ids, current], dim=1)
            attn_mask = torch.cat(
                [condition_mask, torch.ones(B, seq_length, device=device, dtype=torch.long)],
                dim=1,
            )

            # Forward pass through LLaDA (bidirectional)
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, L_c:, :]  # [B, seq_length, V]

            # Add Gumbel noise for sampling diversity
            if temperature > 0:
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits).clamp(min=1e-10)
                ))
                perturbed_logits = logits / temperature + gumbel_noise
            else:
                perturbed_logits = logits

            # Get predicted tokens (argmax of perturbed logits)
            predicted = perturbed_logits.argmax(dim=-1)  # [B, seq_length]

            # Confidence: softmax probability of the predicted token
            probs = F.softmax(logits, dim=-1)  # [B, seq_length, V]
            confidences = probs.gather(2, predicted.unsqueeze(-1)).squeeze(-1)  # [B, seq_length]

            # Only consider currently masked positions for unmasking
            confidences[is_unmasked] = -float("inf")

            # Select top-k most confident masked positions to unmask
            if num_to_unmask > 0:
                remaining_masked = (~is_unmasked).sum(dim=-1).min().item()
                k = min(num_to_unmask, int(remaining_masked))
                if k > 0:
                    _, top_indices = confidences.topk(k, dim=-1)  # [B, k]

                    # Unmask selected positions
                    for b in range(B):
                        for idx in top_indices[b]:
                            current[b, idx] = predicted[b, idx]
                            is_unmasked[b, idx] = True

            # Remask non-finalized positions (they stay as mask token)
            # current already has mask tokens at non-unmasked positions

        # Final pass: predict any remaining masked positions
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
