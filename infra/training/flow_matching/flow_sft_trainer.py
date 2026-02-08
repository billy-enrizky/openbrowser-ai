"""Flow SFT trainer: Conditional Flow Matching loss training."""

import json
import logging

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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

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
            # Simplified: in practice, tokenize condition and target properly
            # This is a skeleton -- actual implementation needs proper tokenization
            global_step += 1

            optimizer.zero_grad()
            # Placeholder loss computation -- requires proper batch tokenization
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if global_step % config["logging_steps"] == 0:
                logger.info(f"Epoch {epoch+1}, Step {global_step}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), "outputs/flow_matching_sft/model.pt")
    logger.info("Flow SFT training complete")


if __name__ == "__main__":
    train()
