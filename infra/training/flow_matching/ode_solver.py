"""ODE solver for discrete flow matching inference."""

import logging

import torch

logger = logging.getLogger(__name__)


def euler_solve(
    model,
    noise: torch.Tensor,
    condition: torch.Tensor,
    num_steps: int = 20,
    sigma_min: float = 0.001,
) -> torch.Tensor:
    """Euler method ODE solver for flow matching.

    Integrates dx/dt = v_theta(x_t, t, c) from t=0 (noise) to t=1 (data).

    Args:
        model: FlowVectorFieldEstimator
        noise: [B, L] initial noise token ids
        condition: [B, L_c] condition token ids
        num_steps: number of Euler steps
        sigma_min: minimum noise scale

    Returns:
        [B, L] predicted token ids at t=1
    """
    device = noise.device
    dt = 1.0 / num_steps

    # Start from noise (t=0)
    x_t = noise.float()
    vocab_size = model.vocab_size

    for step in range(num_steps):
        t = torch.full((noise.shape[0],), step * dt, device=device)

        # Get velocity
        with torch.no_grad():
            # Need integer tokens for embedding lookup
            x_t_ids = x_t.long().clamp(0, vocab_size - 1)
            velocity = model(x_t_ids, t, condition)

        # Euler step in embedding space
        # Project velocity back to token logits
        token_emb = model.token_embedding.weight  # [V, D]
        logits = torch.matmul(velocity, token_emb.T)  # [B, L, V]

        # Soft update: mix current with predicted
        current_one_hot = torch.nn.functional.one_hot(
            x_t_ids, num_classes=vocab_size
        ).float()
        predicted_probs = torch.softmax(logits, dim=-1)

        # Interpolate
        alpha = dt
        mixed = (1 - alpha) * current_one_hot + alpha * predicted_probs

        # Sample or argmax
        x_t = mixed.argmax(dim=-1).float()

    return x_t.long()


def sample(
    model,
    condition: torch.Tensor,
    seq_length: int,
    num_steps: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from the flow model.

    Args:
        model: FlowVectorFieldEstimator
        condition: [B, L_c] condition token ids
        seq_length: output sequence length
        num_steps: number of ODE steps
        temperature: sampling temperature (1.0 = no change)

    Returns:
        [B, seq_length] generated token ids
    """
    B = condition.shape[0]
    device = condition.device

    # Start from uniform random noise tokens
    noise = torch.randint(0, model.vocab_size, (B, seq_length), device=device)

    # Solve ODE
    output = euler_solve(model, noise, condition, num_steps=num_steps)

    return output
