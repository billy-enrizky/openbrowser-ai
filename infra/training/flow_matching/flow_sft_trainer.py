"""Flow SFT trainer: Conditional Flow Matching loss training."""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from infra.training.flow_matching.config import FLOW_SFT_CONFIG, DATA_CONFIG, FLOW_MODEL_CONFIG
from infra.training.flow_matching.flow_model import FlowVectorFieldEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowDataset(Dataset):
    """Dataset for flow matching training."""

    def __init__(self, file_path: str, max_samples: int = 0):
        self.data = []
        with open(file_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        if max_samples > 0:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} flow training examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize_for_flow(
    texts: list[str], max_length: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Simple hash-based tokenization for flow matching.

    Maps each byte to a token ID in [0, vocab_size). In production,
    this should use a proper tokenizer aligned with the vocabulary.
    """
    batch_ids = []
    for text in texts:
        tokens = text.encode("utf-8")
        ids = [b % vocab_size for b in tokens][:max_length]
        ids = ids + [0] * (max_length - len(ids))
        batch_ids.append(ids)
    return torch.tensor(batch_ids, dtype=torch.long, device=device)


def flow_collate_fn(batch):
    """Return batch as a list of dicts (skip default collation)."""
    return batch


def cfm_loss(
    model: FlowVectorFieldEstimator,
    x_0: torch.Tensor,  # noise tokens [B, L]
    x_1: torch.Tensor,  # target tokens [B, L]
    condition: torch.Tensor,  # condition tokens [B, L_c]
    sigma_min: float = 0.001,
) -> torch.Tensor:
    """Conditional Flow Matching loss.

    L = E_t,x_0 || v_theta(x_t, t, c) - (x_1 - x_0) ||^2

    where x_t = (1-t) * x_0 + t * x_1 (linear interpolation).
    """
    B = x_0.shape[0]
    device = x_0.device

    # Sample random time
    t = torch.rand(B, device=device)

    # Linear interpolation in token space (using embeddings)
    # For discrete tokens, we interpolate the embeddings
    x_0_emb = model.token_embedding(x_0)  # [B, L, D]
    x_1_emb = model.token_embedding(x_1)  # [B, L, D]

    t_expanded = t.view(B, 1, 1)
    x_t_emb = (1 - t_expanded) * x_0_emb + t_expanded * x_1_emb

    # Target velocity: x_1 - x_0 in embedding space
    target_velocity = x_1_emb - x_0_emb

    # Predicted velocity (use nearest token ids for model input)
    logits = torch.matmul(x_t_emb, model.token_embedding.weight.T)
    x_t_ids = logits.argmax(dim=-1)

    predicted_velocity = model(x_t_ids, t, condition)

    # MSE loss
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    return loss


def train():
    """Run flow SFT training."""
    config = FLOW_SFT_CONFIG

    model = FlowVectorFieldEstimator(FLOW_MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = FlowDataset(DATA_CONFIG["train_file"], max_samples=DATA_CONFIG.get("max_train_samples", 0))
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=flow_collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    logger.info(f"Starting flow SFT training on {device}")
    logger.info(f"Dataset size: {len(dataset)}, Epochs: {config['num_epochs']}")

    global_step = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            global_step += 1

            # Extract condition and target text from batch items
            conditions = [item.get("condition", "") for item in batch]
            targets = [
                " ".join(t) if isinstance(t, list) else str(t)
                for t in (item.get("target", []) for item in batch)
            ]

            # Tokenize for flow model
            vocab_size = model.vocab_size
            max_seq_length = model.max_seq_length
            condition_ids = tokenize_for_flow(conditions, max_seq_length, vocab_size, device)
            target_ids = tokenize_for_flow(targets, max_seq_length, vocab_size, device)

            # Sample noise tokens
            x_0 = torch.randint(0, vocab_size, target_ids.shape, device=device)

            optimizer.zero_grad()
            loss = cfm_loss(
                model, x_0, target_ids, condition_ids,
                sigma_min=config.get("sigma_min", 0.001),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if global_step % config["logging_steps"] == 0:
                logger.info(f"Epoch {epoch+1}, Step {global_step}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

    # Save
    output_dir = Path("outputs/flow_matching_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_dir / "model.pt"))
    logger.info("Flow SFT training complete")


if __name__ == "__main__":
    train()
