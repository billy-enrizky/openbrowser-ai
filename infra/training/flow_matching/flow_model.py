"""Transformer-based vector field estimator for discrete flow matching (~100M params)."""

import logging
import math

import torch
import torch.nn as nn

from infra.training.flow_matching.config import FLOW_MODEL_CONFIG

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FlowTransformerBlock(nn.Module):
    """Transformer block with time-conditioned layer norm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        # Time modulation
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Time modulation: scale and shift
        time_out = self.time_mlp(t_emb).unsqueeze(1)
        scale, shift = time_out.chunk(2, dim=-1)

        # Self-attention with time conditioning
        h = self.norm1(x) * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FlowVectorFieldEstimator(nn.Module):
    """Transformer vector field v_theta(x_t, t, c) for discrete flow matching.

    Predicts the velocity field that transports noise to data given:
    - x_t: noised token embeddings at time t
    - t: time in [0, 1]
    - c: condition (instruction) embeddings
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or FLOW_MODEL_CONFIG

        self.vocab_size = config["vocab_size"]
        self.hidden_dim = config["hidden_dim"]
        self.max_seq_length = config["max_seq_length"]

        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_dim)

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(self.hidden_dim)
        self.time_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Condition encoder (simplified -- uses same embedding)
        self.condition_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(self.hidden_dim, config["num_heads"], config["dropout"])
            for _ in range(config["num_layers"])
        ])

        self.norm = nn.LayerNorm(self.hidden_dim)

        # Output: predict velocity in embedding space
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"FlowVectorFieldEstimator: {total_params / 1e6:.1f}M parameters")

    def forward(
        self,
        x_t: torch.Tensor,       # [B, L] token ids at time t
        t: torch.Tensor,          # [B] time values in [0, 1]
        condition: torch.Tensor,  # [B, L_c] condition token ids
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, c).

        Returns: [B, L, hidden_dim] velocity vectors in embedding space.
        """
        B, L = x_t.shape

        # Embed tokens
        positions = torch.arange(L, device=x_t.device).unsqueeze(0)
        x = self.token_embedding(x_t) + self.position_embedding(positions)

        # Embed time
        t_emb = self.time_embedding(t)
        t_emb = self.time_proj(t_emb)

        # Embed condition and add to input
        c_emb = self.token_embedding(condition)
        c_emb = self.condition_proj(c_emb.mean(dim=1))  # Pool condition
        x = x + c_emb.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        x = self.norm(x)
        velocity = self.output_proj(x)

        return velocity
