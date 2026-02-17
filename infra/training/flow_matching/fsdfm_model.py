"""FS-DFM 1.3B: True Discrete Flow Matching model with LoRA fine-tuning.

Self-contained implementation of Apple's FS-DFM DiT architecture (Diffusion
Transformer) with flow matching utilities. No dependency on Apple's repo.

Architecture (1.3B config) -- matches Apple's exact naming convention:
    - DDiTBlock: adaLN modulation + multi-head self-attention (qw/kw/vw) + MLP
    - Rotary positional embeddings (RoPE)
    - TimestepEmbedder: sinusoidal frequency + 2-layer MLP -> cond_dim
    - DDitFinalLayer (output_layer): adaLN + linear projection to vocab logits
    - GPT-2 tokenizer (vocab_size=50257), max_seq_length=1024

Flow matching:
    - PolynomialConvexScheduler: alpha_t = t^n, sigma_t = 1 - t^n
    - MixtureDiscreteProbPath: P(X_t=X_0) = sigma_t, P(X_t=X_1) = 1-sigma_t
    - Generalized KL loss for discrete CTMC flow matching
    - Discrete Euler solver with edit_mask for prefix-conditioned generation

References:
    - Config: hidden_size=2048, n_blocks=21, n_heads=32, cond_dim=256
    - Checkpoint: aminr8/FS-DFM -> DFM_checkpoint.pth
    - Paper: "Simplified and Generalized Masked Diffusion for Discrete Data"
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


def precompute_rotary_freqs(dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_len, dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor."""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)  # [1, 1, L, D]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Adaptive LayerNorm helpers
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Model components (matching Apple's exact naming convention)
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embed scalar timestep t into a conditioning vector via sinusoidal + MLP.

    Output dimension is cond_dim (256), not hidden_size.
    """

    def __init__(self, cond_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar timesteps in [0, 1] -> [B, cond_dim]."""
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Cast to model dtype (sinusoidal math produces float32, weights may be bfloat16)
        embedding = embedding.to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class DDiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive LayerNorm (adaLN).

    Matches Apple's FS-DFM checkpoint naming:
        - Separate qw, kw, vw projections (not packed attn_qkv)
        - mlp = nn.Sequential(Linear, GELU, Linear) with bias=True
        - adaLN_modulation = bare nn.Linear (SiLU applied in forward)
        - norm1/norm2 = LayerNorm without bias
    """

    def __init__(self, hidden_size: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Layer norms (weight only, no bias -- matches checkpoint)
        self.norm1 = nn.LayerNorm(hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(hidden_size, bias=False)

        # Self-attention: separate Q, K, V projections (no bias)
        self.qw = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kw = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vw = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP as Sequential (matches checkpoint keys: mlp.0.weight, mlp.0.bias, mlp.2.weight, mlp.2.bias)
        mlp_hidden = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

        # adaLN modulation: bare Linear (SiLU applied in forward, not wrapped in Sequential)
        # Checkpoint key: adaLN_modulation.weight (not adaLN_modulation.1.weight)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, H] hidden states
        c: [B, cond_dim] conditioning vector (from timestep)
        rotary_cos, rotary_sin: precomputed RoPE tables
        """
        B, L, H = x.shape

        # adaLN modulation (SiLU applied here, not stored in the module)
        mod = self.adaLN_modulation(F.silu(c)).unsqueeze(1)  # [B, 1, 6*H]
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Norm + modulate + attention
        h = modulate(self.norm1(x), shift1, scale1)

        # Separate Q, K, V projections
        q = self.qw(h)
        k = self.kw(h)
        v = self.vw(h)

        # Reshape for multi-head attention
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        q = apply_rotary_emb(q, rotary_cos, rotary_sin)
        k = apply_rotary_emb(k, rotary_cos, rotary_sin)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.1)
        attn = attn.transpose(1, 2).contiguous().view(B, L, H)

        # Output projection + gated residual
        attn = self.attn_out(attn)
        x = x + gate1 * self.dropout(attn)

        # Norm + modulate + MLP
        h = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h)
        x = x + gate2 * self.dropout(mlp_out)

        return x


class DDitFinalLayer(nn.Module):
    """Final layer: adaLN + linear projection to vocabulary logits.

    Matches Apple's checkpoint naming:
        - norm_final = LayerNorm (no bias, same as block norms)
        - adaLN_modulation = bare nn.Linear
        - linear = nn.Linear with bias=True
    """

    def __init__(self, hidden_size: int, cond_dim: int, vocab_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, bias=False)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(F.silu(c)).unsqueeze(1)
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FSDFMTransformer(nn.Module):
    """FS-DFM Diffusion Transformer (DiT) for discrete flow matching.

    Forward: x_t (noised tokens) + t (timestep) -> logits [B, L, vocab_size]

    Attribute names match Apple's checkpoint exactly:
        - vocab_embed (not tok_emb), size = vocab_size (50257)
        - time_embedding (not time_embedder)
        - blocks (same)
        - output_layer (not final_layer)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        n_blocks = config["n_blocks"]
        n_heads = config["n_heads"]
        cond_dim = config["cond_dim"]
        mlp_ratio = config.get("mlp_ratio", 4)
        vocab_size = config["vocab_size"]
        max_seq_length = config["max_seq_length"]
        dropout = config.get("dropout", 0.1)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        # Token embedding (GPT-2 vocab, 50257 tokens)
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)

        # Timestep conditioning -> cond_dim output
        self.time_embedding = TimestepEmbedder(cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        # Final layer -> logits
        self.output_layer = DDitFinalLayer(hidden_size, cond_dim, vocab_size)

        # Precompute rotary embeddings (not in state_dict, persistent=False)
        head_dim = hidden_size // n_heads
        rotary_cos, rotary_sin = precompute_rotary_freqs(head_dim, max_seq_length)
        self.register_buffer("rotary_cos", rotary_cos, persistent=False)
        self.register_buffer("rotary_sin", rotary_sin, persistent=False)

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"FSDFMTransformer: {total_params / 1e9:.2f}B parameters")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, L] noised token IDs at time t
            t: [B] timestep values in [0, 1]

        Returns:
            logits: [B, L, vocab_size] predicted posterior distribution
        """
        # Token embeddings
        x = self.vocab_embed(x_t)  # [B, L, H]

        # Timestep conditioning
        c = self.time_embedding(t)  # [B, cond_dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, self.rotary_cos, self.rotary_sin)

        # Final layer -> logits
        logits = self.output_layer(x, c)  # [B, L, V]
        return logits


# ---------------------------------------------------------------------------
# LoRA: Low-Rank Adaptation for fine-tuning
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with frozen base + trainable LoRA."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        # Create LoRA params on the same device/dtype as the base weights
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_features, r, device=device, dtype=dtype) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model: FSDFMTransformer, config: dict) -> FSDFMTransformer:
    """Replace attention projection layers with LoRA variants.

    Targets separate qw, kw, vw, attn_out (matching Apple's naming).
    Freezes all base parameters, only LoRA A/B matrices are trainable.
    """
    r = config.get("lora_r", 16)
    alpha = config.get("lora_alpha", 32)
    target_layers = config.get("lora_target_layers", ["qw", "kw", "vw", "attn_out"])

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    injected = 0
    for block in model.blocks:
        for layer_name in target_layers:
            if hasattr(block, layer_name):
                original = getattr(block, layer_name)
                lora_layer = LoRALinear(original, r=r, alpha=alpha)
                setattr(block, layer_name, lora_layer)
                injected += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA injected into {injected} layers: "
        f"{trainable / 1e6:.2f}M trainable / {total / 1e6:.1f}M total "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def save_lora_weights(model: FSDFMTransformer, path: str):
    """Save only LoRA adapter weights (~22MB)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_state[name] = param.data.cpu()

    torch.save(lora_state, path)
    logger.info(f"LoRA weights saved to {path} ({len(lora_state)} tensors)")


def load_lora_weights(model: FSDFMTransformer, path: str):
    """Load LoRA adapter weights into an already-injected model."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()

    loaded = 0
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded += 1
        else:
            logger.warning(f"LoRA weight {name} not found in model")

    logger.info(f"Loaded {loaded}/{len(lora_state)} LoRA weights from {path}")


# ---------------------------------------------------------------------------
# Checkpoint loading from HuggingFace
# ---------------------------------------------------------------------------


def _extract_model_state_dict(raw: dict) -> dict:
    """Extract model weights from a checkpoint that may contain optimizer state.

    Apple's DFM_checkpoint.pth is a nested dict with keys like
    ['optimizer', 'model', 'step', 'train_sampler', 'test_sampler'].
    This function finds and returns only the model weights.
    """
    # If it looks like a flat state_dict (keys contain weight/bias tensor names)
    sample_keys = list(raw.keys())[:5]
    if any("." in k for k in sample_keys):
        return raw

    # Check for common nested keys
    for key in ["model_state_dict", "state_dict", "model", "module"]:
        if key in raw:
            inner = raw[key]
            if isinstance(inner, dict):
                logger.info(f"Extracted model weights from nested key '{key}'")
                return inner

    # If 'model' key not found but optimizer-like keys exist, filter them out
    model_keys = {}
    skip_prefixes = ("optimizer", "scheduler", "scaler", "lr_scheduler", "step", "epoch")
    for k, v in raw.items():
        if not any(k.startswith(sp) or k == sp for sp in skip_prefixes):
            if isinstance(v, torch.Tensor):
                model_keys[k] = v
    if model_keys:
        logger.info(
            f"Filtered {len(model_keys)} model tensors from {len(raw)} total keys"
        )
        return model_keys

    return raw


def load_fsdfm_from_huggingface(
    config: dict,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> FSDFMTransformer:
    """Download DFM_checkpoint.pth from HuggingFace and load into FSDFMTransformer.

    The checkpoint is ~16GB (FP32 model + optimizer state). We use mmap=True
    to keep tensor data on disk, extract only model weights, convert to target
    dtype, then load into the model. Peak CPU RAM usage is ~8GB.
    """
    import gc

    from huggingface_hub import hf_hub_download

    repo_id = config["hf_repo"]
    filename = config["checkpoint_filename"]

    logger.info(f"Downloading {filename} from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"Checkpoint downloaded to {ckpt_path}")

    # Memory-mapped load: tensor data stays on disk, not in RAM
    logger.info("Loading checkpoint with mmap (data stays on disk)...")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    logger.info(f"Checkpoint loaded (mmap). Top-level keys: {list(raw_ckpt.keys())[:10]}")

    # Extract only model weights (skip optimizer state, etc.)
    state_dict = _extract_model_state_dict(raw_ckpt)

    # Handle potential key prefix mismatches
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Convert model weights to target dtype. Since source tensors are mmap'd
    # (on disk), .to(dtype) creates new tensors in RAM at the target dtype size.
    logger.info(f"Converting {len(state_dict)} model tensors to {dtype}...")
    converted = {}
    for key in state_dict:
        v = state_dict[key]
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            converted[key] = v.to(dtype)
        elif isinstance(v, torch.Tensor):
            converted[key] = v.clone()
        else:
            converted[key] = v

    # Free the mmap'd checkpoint
    del raw_ckpt, state_dict
    gc.collect()
    logger.info(f"State dict converted to {dtype} ({len(converted)} tensors)")

    # Log some checkpoint keys for debugging
    ckpt_keys = sorted(converted.keys())
    logger.info(f"Checkpoint keys (first 10): {ckpt_keys[:10]}")
    logger.info(f"Checkpoint keys (last 10): {ckpt_keys[-10:]}")

    # Create model on CPU (default dtype float32, params will be replaced by assign=True)
    logger.info("Creating FSDFMTransformer on CPU...")
    model = FSDFMTransformer(config)

    # Log model state dict keys for comparison
    model_keys = sorted(model.state_dict().keys())
    logger.info(f"Model keys (first 10): {model_keys[:10]}")
    logger.info(f"Model keys (last 10): {model_keys[-10:]}")

    # Remove checkpoint-only keys that we recompute (rotary embeddings)
    model_key_set = set(model.state_dict().keys())
    extra_keys = [k for k in converted if k not in model_key_set]
    for k in extra_keys:
        logger.info(f"Dropping checkpoint-only key (recomputed locally): {k}")
        del converted[k]

    # Load checkpoint weights with assign=True (replaces parameters, memory efficient)
    missing, unexpected = model.load_state_dict(converted, assign=True, strict=False)
    if missing:
        logger.error(f"MISSING keys ({len(missing)}): {missing}")
    if unexpected:
        logger.error(f"UNEXPECTED keys ({len(unexpected)}): {list(unexpected)}")
    if not missing and not unexpected:
        logger.info("All checkpoint keys matched model keys perfectly")

    del converted
    gc.collect()

    # Move to target device
    model = model.to(device=device)
    logger.info(f"FS-DFM model loaded: {dtype} on {device}")
    return model


# ---------------------------------------------------------------------------
# Flow Matching Utilities
# ---------------------------------------------------------------------------


class PolynomialConvexScheduler:
    """Polynomial scheduler: alpha_t = t^n, sigma_t = 1 - t^n.

    d_alpha_t/dt = n * t^(n-1)
    """

    def __init__(self, exponent: float = 2.0):
        self.n = exponent

    def __call__(self, t: torch.Tensor) -> dict:
        """Compute scheduler values at time t.

        Returns dict with alpha_t, sigma_t, d_alpha_t (derivative).
        """
        alpha_t = t.pow(self.n)
        sigma_t = 1.0 - alpha_t
        d_alpha_t = self.n * t.pow(self.n - 1)
        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "d_alpha_t": d_alpha_t,
        }


@dataclass
class DiscretePathSample:
    """Sample from discrete probability path."""
    x_t: torch.Tensor
    x_0: torch.Tensor
    x_1: torch.Tensor
    t: torch.Tensor


class MixtureDiscreteProbPath:
    """Discrete mixture probability path for flow matching.

    P(X_t = X_0) = sigma_t = 1 - t^n   (source / noise)
    P(X_t = X_1) = alpha_t = t^n        (target / data)

    At t=0: X_t = X_0 (pure noise)
    At t=1: X_t = X_1 (clean data)
    """

    def __init__(self, scheduler: PolynomialConvexScheduler):
        self.scheduler = scheduler

    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> DiscretePathSample:
        """Sample x_t from the mixture path.

        x_0: [B, L] source (noise) tokens
        x_1: [B, L] target (clean) tokens
        t: [B] or scalar timestep

        Returns DiscretePathSample with x_t where each token is independently
        chosen from x_0 with prob sigma_t or x_1 with prob alpha_t.
        """
        sched = self.scheduler(t)
        sigma_t = sched["sigma_t"]

        # Expand for broadcasting: [B, 1] for per-token sampling
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.unsqueeze(-1)

        # Bernoulli: each position independently picks source or target
        source_mask = torch.rand_like(x_0.float()) < sigma_t
        x_t = torch.where(source_mask, x_0, x_1)

        return DiscretePathSample(x_t=x_t, x_0=x_0, x_1=x_1, t=t)


def compute_generalized_kl_loss(
    logits: torch.Tensor,
    x_1: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    scheduler: PolynomialConvexScheduler,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized KL loss for discrete flow matching.

    loss = -jump_coeff * [p_1t(x_t) - delta + (1 - delta) * log_p_1t(x_1)]

    where:
        - p_1t = softmax(logits): predicted posterior over clean tokens
        - delta = 1 if x_1 == x_t else 0
        - jump_coeff = d_alpha_t / (1 - alpha_t): rate of the jump process

    Args:
        logits: [B, L, V] model output logits
        x_1: [B, L] target (clean) token IDs
        x_t: [B, L] noised token IDs at time t
        t: [B] timestep values
        scheduler: PolynomialConvexScheduler instance
        loss_mask: [B, L] optional mask (1 = compute loss, 0 = ignore)

    Returns:
        Scalar loss tensor.
    """
    sched = scheduler(t)
    alpha_t = sched["alpha_t"]
    d_alpha_t = sched["d_alpha_t"]

    # Jump coefficient: d_alpha_t / (1 - alpha_t), with clamp for stability
    jump_coeff = d_alpha_t / (1.0 - alpha_t).clamp(min=1e-6)
    jump_coeff = jump_coeff.unsqueeze(-1)  # [B, 1]

    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
    probs = torch.exp(log_probs)

    # Extract p_1t(x_1) and p_1t(x_t)
    log_p_x1 = log_probs.gather(2, x_1.unsqueeze(-1)).squeeze(-1)  # [B, L]
    p_xt = probs.gather(2, x_t.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # Delta indicator: 1 where x_1 == x_t
    delta = (x_1 == x_t).float()

    # Generalized KL loss per token
    per_token_loss = -jump_coeff * (p_xt - delta + (1.0 - delta) * log_p_x1)

    # Apply loss mask if provided
    if loss_mask is not None:
        per_token_loss = per_token_loss * loss_mask
        return per_token_loss.sum() / loss_mask.sum().clamp(min=1)

    return per_token_loss.mean()


# ---------------------------------------------------------------------------
# Discrete Euler Solver (Generation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def discrete_euler_solve(
    model: FSDFMTransformer,
    x_init: torch.Tensor,
    num_steps: int,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    temperature: float = 1.0,
    edit_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discrete Euler solver for generation via Poisson jump process.

    Starting from noise x_0 at t=0, iteratively denoise toward t=1 (clean data).
    Optionally uses edit_mask for prefix-conditioned generation (inpainting).

    Args:
        model: FSDFMTransformer to call for posterior predictions.
        x_init: [B, L] initial noisy tokens (x_0, uniform random or masked).
        num_steps: Number of Euler steps from t=0 to t=1.
        scheduler: PolynomialConvexScheduler for computing jump rates.
        vocab_size: Size of token vocabulary.
        temperature: Sampling temperature (1.0 = standard, 0 = greedy).
        edit_mask: [B, L] bool tensor. True = editable (generate), False = frozen (keep).
                   If None, all positions are editable.

    Returns:
        x_final: [B, L] generated token IDs.
    """
    model.eval()
    device = x_init.device
    x_t = x_init.clone()
    B, L = x_t.shape

    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)

        # Get model predictions (posterior p_{1|t})
        logits = model(x_t, t)  # [B, L, V]

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)  # [B, L, V]

        # Compute jump rates
        sched = scheduler(t)
        alpha_t_val = sched["alpha_t"]
        d_alpha_t_val = sched["d_alpha_t"]
        rate_scale = (d_alpha_t_val / (1.0 - alpha_t_val).clamp(min=1e-6))
        rate_scale = rate_scale.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        # Jump rates: u_ij = rate_scale * p_{1|t}(j | x_t) for j != x_t[i]
        # Total rate per position: lambda_i = rate_scale * (1 - p_{1|t}(x_t[i]))
        p_current = probs.gather(2, x_t.unsqueeze(-1)).squeeze(-1)  # [B, L]
        total_rate = rate_scale.squeeze(-1) * (1.0 - p_current)  # [B, L]

        # Poisson jump probability: P(jump) = 1 - exp(-lambda * dt)
        jump_prob = 1.0 - torch.exp(-total_rate * dt)

        # Sample which positions jump
        jumps = torch.rand_like(jump_prob) < jump_prob  # [B, L]

        # Apply edit mask: only editable positions can jump
        if edit_mask is not None:
            jumps = jumps & edit_mask

        # For jumping positions, sample new token from posterior
        # Zero out current token probability for off-diagonal sampling
        current_one_hot = F.one_hot(x_t, vocab_size).float()
        off_diag_probs = (probs - current_one_hot * probs).clamp(min=0)
        off_diag_probs = off_diag_probs / off_diag_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Sample new tokens
        new_tokens = torch.multinomial(
            off_diag_probs.view(-1, vocab_size), num_samples=1
        ).view(B, L)

        # Apply jumps
        x_t = torch.where(jumps, new_tokens, x_t)

    # Final step: collapse to most likely token at remaining noisy positions
    t_final = torch.ones(B, device=device, dtype=torch.float32)
    final_logits = model(x_t, t_final)
    if temperature != 1.0 and temperature > 0:
        final_logits = final_logits / temperature
    final_tokens = final_logits.argmax(dim=-1)

    if edit_mask is not None:
        x_t = torch.where(edit_mask, final_tokens, x_t)
    else:
        x_t = final_tokens

    return x_t


def generate_with_prefix_conditioning(
    model: FSDFMTransformer,
    prefix_ids: torch.Tensor,
    gen_length: int,
    config: dict,
    scheduler: PolynomialConvexScheduler,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate response tokens conditioned on a fixed prefix (instruction).

    Concatenates prefix + random noise, then runs the Euler solver with
    an edit_mask that freezes the prefix and only denoises the response.

    Args:
        model: FSDFMTransformer model.
        prefix_ids: [B, L_prefix] instruction token IDs.
        gen_length: Number of response tokens to generate.
        config: Model config dict.
        scheduler: PolynomialConvexScheduler instance.
        temperature: Sampling temperature.

    Returns:
        generated_ids: [B, gen_length] generated response token IDs.
    """
    B = prefix_ids.shape[0]
    L_prefix = prefix_ids.shape[1]
    vocab_size = config["vocab_size"]
    device = prefix_ids.device

    # Initialize: prefix (fixed) + noise (uniform random tokens for response)
    noise = torch.randint(0, vocab_size, (B, gen_length), device=device)
    x_init = torch.cat([prefix_ids, noise], dim=1)  # [B, L_prefix + gen_length]

    # Edit mask: False for prefix (frozen), True for response (editable)
    edit_mask = torch.zeros(B, L_prefix + gen_length, dtype=torch.bool, device=device)
    edit_mask[:, L_prefix:] = True

    num_steps = config.get("num_sampling_steps", 64)

    # Run Euler solver
    x_final = discrete_euler_solve(
        model=model,
        x_init=x_init,
        num_steps=num_steps,
        scheduler=scheduler,
        vocab_size=vocab_size,
        temperature=temperature,
        edit_mask=edit_mask,
    )

    # Extract generated response tokens
    return x_final[:, L_prefix:]


# ---------------------------------------------------------------------------
# Flow-GRPO: Trajectory-Recording Solver and Per-Step Log-Probability
# ---------------------------------------------------------------------------
# Adapted from Flow-GRPO (Liu et al., 2025) for discrete flow matching.
# Reference: github.com/yifan123/flow_grpo
#
# In continuous Flow-GRPO, each SDE step gives a Gaussian policy whose
# log-prob is a Gaussian log-density.  In our discrete adaptation, each
# Euler step of the Poisson jump CTMC gives a categorical distribution
# per position, whose log-prob we compute exactly.
# ---------------------------------------------------------------------------


@dataclass
class EulerTrajectoryStep:
    """Recorded data from one discrete Euler step for policy gradient computation.

    Stores only token IDs (lightweight); model logits/probs are recomputed
    during training to allow gradient flow through the current policy.
    """
    t_value: float            # Timestep scalar at the start of this step
    x_t: torch.Tensor         # [B, L] token IDs before this transition
    x_next: torch.Tensor      # [B, L] token IDs after this transition


@dataclass
class EulerTrajectory:
    """Full generation trajectory for Flow-GRPO policy gradient computation."""
    steps: list[EulerTrajectoryStep]
    final_tokens: torch.Tensor   # [B, L] final generated tokens
    edit_mask: torch.Tensor      # [B, L] bool -- True for response (editable) positions


@torch.no_grad()
def discrete_euler_solve_with_trajectory(
    model: FSDFMTransformer,
    x_init: torch.Tensor,
    num_steps: int,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    temperature: float = 1.0,
    edit_mask: torch.Tensor | None = None,
) -> EulerTrajectory:
    """Discrete Euler solver that records the full trajectory for Flow-GRPO.

    Same Poisson jump logic as ``discrete_euler_solve`` but additionally
    stores (x_t, x_next) at each step so that per-step log-probabilities
    can be recomputed during training with gradient flow.

    Memory overhead per trajectory: ~2 * T * B * L * 8 bytes (int64 tensors).
    For T=10, B=1, L=1024 this is ~160 KB -- negligible.

    Args:
        model: FSDFMTransformer to call for posterior predictions.
        x_init: [B, L] initial noisy tokens (uniform random for response).
        num_steps: Number of Euler steps from t=0 to t=1.
        scheduler: PolynomialConvexScheduler instance.
        vocab_size: Token vocabulary size.
        temperature: Sampling temperature (1.0 = standard).
        edit_mask: [B, L] bool. True = editable (generate), False = frozen.

    Returns:
        EulerTrajectory with recorded steps and final tokens.
    """
    model.eval()
    device = x_init.device
    x_t = x_init.clone()
    B, L = x_t.shape
    dt = 1.0 / num_steps

    trajectory_steps: list[EulerTrajectoryStep] = []

    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)

        x_before = x_t.clone()

        # Model posterior p_{1|t}
        logits = model(x_t, t)  # [B, L, V]
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Jump rates
        sched = scheduler(t)
        alpha_t_val = sched["alpha_t"]
        d_alpha_t_val = sched["d_alpha_t"]
        rate_scale = d_alpha_t_val / (1.0 - alpha_t_val).clamp(min=1e-6)
        rate_scale = rate_scale.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        p_current = probs.gather(2, x_t.unsqueeze(-1)).squeeze(-1)  # [B, L]
        total_rate = rate_scale.squeeze(-1) * (1.0 - p_current)  # [B, L]

        # Poisson jump probability
        jump_prob = 1.0 - torch.exp(-total_rate * dt)
        jumps = torch.rand_like(jump_prob) < jump_prob

        # Apply edit mask
        if edit_mask is not None:
            jumps = jumps & edit_mask

        # Sample new tokens from off-diagonal posterior
        current_one_hot = F.one_hot(x_t, vocab_size).float()
        off_diag_probs = (probs - current_one_hot * probs).clamp(min=0)
        off_diag_probs = off_diag_probs / off_diag_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        new_tokens = torch.multinomial(
            off_diag_probs.view(-1, vocab_size), num_samples=1
        ).view(B, L)

        x_t = torch.where(jumps, new_tokens, x_t)

        trajectory_steps.append(EulerTrajectoryStep(
            t_value=t_val,
            x_t=x_before,
            x_next=x_t.clone(),
        ))

    # Final step: collapse to most likely token
    t_final = torch.ones(B, device=device, dtype=torch.float32)
    final_logits = model(x_t, t_final)
    if temperature != 1.0 and temperature > 0:
        final_logits = final_logits / temperature
    final_tokens = final_logits.argmax(dim=-1)

    if edit_mask is not None:
        final_tokens = torch.where(edit_mask, final_tokens, x_t)

    # Record the final collapse as the last step
    trajectory_steps.append(EulerTrajectoryStep(
        t_value=1.0 - dt,  # Last regular step time
        x_t=x_t.clone(),
        x_next=final_tokens.clone(),
    ))

    if edit_mask is None:
        edit_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    return EulerTrajectory(
        steps=trajectory_steps,
        final_tokens=final_tokens,
        edit_mask=edit_mask,
    )


def compute_discrete_step_log_prob(
    model: FSDFMTransformer,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t_scalar: float,
    dt: float,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute log P(x_next | x_t) under the discrete Poisson jump process.

    This is the discrete analog of ``sde_step_with_logprob`` from
    continuous Flow-GRPO: it computes the log-probability of an observed
    transition under the model's posterior, enabling REINFORCE / PPO
    policy gradients aligned with the actual generation process.

    Per-position transition distribution (for response tokens):
        P(stay at x_t[i]) = exp(-lambda_i * dt)
        P(jump to j != x_t[i]) = (1 - exp(-lambda_i * dt)) * p(j) / (1 - p(x_t[i]))
    where lambda_i = rate_scale * (1 - p(x_t[i])).

    Args:
        model: FSDFMTransformer (current policy or reference).
        x_t: [B, L] token IDs before transition.
        x_next: [B, L] token IDs after transition.
        t_scalar: Timestep value (float in [0, 1]).
        dt: Step size (1 / num_steps).
        scheduler: PolynomialConvexScheduler.
        vocab_size: Token vocabulary size.
        response_mask: [B, L] float, 1 for response tokens, 0 for prefix/padding.

    Returns:
        log_prob: [B] mean-normalized per-step log-probability (summed over
            response positions, divided by number of response tokens).
    """
    B, L = x_t.shape
    device = x_t.device

    t = torch.full((B,), t_scalar, device=device, dtype=torch.float32)

    # Forward pass (gradients flow if model is in train mode)
    logits = model(x_t, t)  # [B, L, V]
    probs = F.softmax(logits, dim=-1)  # [B, L, V]
    del logits  # Free [B, L, V] logits immediately

    # Extract the two probability values we need, then free the full probs tensor
    p_current = probs.gather(2, x_t.unsqueeze(-1)).squeeze(-1)  # [B, L]
    p_jumped_to = probs.gather(2, x_next.unsqueeze(-1)).squeeze(-1)  # [B, L]
    del probs  # Free [B, L, V] probs -- largest intermediate tensor

    # Jump rates
    sched = scheduler(t)
    alpha_t = sched["alpha_t"]
    d_alpha_t = sched["d_alpha_t"]
    rate_scale = d_alpha_t / (1.0 - alpha_t).clamp(min=1e-6)  # [B]

    # Per-position rate: lambda_i = rate_scale * (1 - p_current)
    lambda_i = rate_scale.unsqueeze(-1) * (1.0 - p_current)  # [B, L]

    # Which positions stayed vs jumped
    stayed = (x_next == x_t)  # [B, L] bool

    # -- Log-prob for positions that stayed --
    # log P(stay) = -lambda_i * dt
    log_prob_stay = -lambda_i * dt  # [B, L]

    # -- Log-prob for positions that jumped --
    # log P(jump to j) = log(1 - exp(-lambda_i * dt)) + log p(j) - log(1 - p(x_t[i]))
    # Use log1p for numerical stability: log(1 - exp(-x)) = log1p(-exp(-x))
    neg_exp_term = torch.exp(-lambda_i * dt)
    log_jump_base = torch.log1p(-neg_exp_term.clamp(max=1.0 - 1e-8))  # [B, L]

    # log p(j) for the actual jumped-to token
    log_p_jumped = torch.log(p_jumped_to.clamp(min=1e-10))  # [B, L]

    # log(1 - p(x_t[i]))
    log_one_minus_p_current = torch.log((1.0 - p_current).clamp(min=1e-8))  # [B, L]

    log_prob_jump = log_jump_base + log_p_jumped - log_one_minus_p_current  # [B, L]

    # Combine: use stay log-prob where stayed, jump log-prob where jumped
    log_prob_per_pos = torch.where(stayed, log_prob_stay, log_prob_jump)  # [B, L]

    # Mask to response tokens and mean-normalize
    log_prob_per_pos = log_prob_per_pos * response_mask
    num_response = response_mask.sum(dim=-1).clamp(min=1)  # [B]
    log_prob = log_prob_per_pos.sum(dim=-1) / num_response  # [B]

    return log_prob


def generate_with_prefix_conditioning_trajectory(
    model: FSDFMTransformer,
    prefix_ids: torch.Tensor,
    gen_length: int,
    config: dict,
    scheduler: PolynomialConvexScheduler,
    temperature: float = 1.0,
) -> EulerTrajectory:
    """Generate response tokens with prefix conditioning, recording trajectory.

    Combines prefix conditioning (instruction fixed, response denoised) with
    trajectory recording for Flow-GRPO policy gradient computation.

    Args:
        model: FSDFMTransformer model.
        prefix_ids: [B, L_prefix] instruction token IDs.
        gen_length: Number of response tokens to generate.
        config: Model config dict (must include vocab_size, num_generation_steps).
        scheduler: PolynomialConvexScheduler instance.
        temperature: Sampling temperature.

    Returns:
        EulerTrajectory with steps, final_tokens, and edit_mask.
        final_tokens are [B, L_prefix + gen_length] (full sequence).
        edit_mask is [B, L_prefix + gen_length] (True for response positions).
    """
    B = prefix_ids.shape[0]
    L_prefix = prefix_ids.shape[1]
    vocab_size = config["vocab_size"]
    device = prefix_ids.device

    # Initialize: prefix (fixed) + noise (uniform random for response)
    noise = torch.randint(0, vocab_size, (B, gen_length), device=device)
    x_init = torch.cat([prefix_ids, noise], dim=1)

    # Edit mask: False for prefix, True for response
    edit_mask = torch.zeros(B, L_prefix + gen_length, dtype=torch.bool, device=device)
    edit_mask[:, L_prefix:] = True

    num_steps = config.get("num_generation_steps", config.get("num_sampling_steps", 64))

    return discrete_euler_solve_with_trajectory(
        model=model,
        x_init=x_init,
        num_steps=num_steps,
        scheduler=scheduler,
        vocab_size=vocab_size,
        temperature=temperature,
        edit_mask=edit_mask,
    )
