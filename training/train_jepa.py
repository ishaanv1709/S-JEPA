"""
Train LeWorldModel (LeWM) — Latent-space pretraining with SIGReg.
Based on: Maes, Le Lidec, Scieur, LeCun, Balestriero (2026)

LeWM training uses ONLY 2 loss terms:
  L_total = L_prediction + lambda * L_SIGReg

  1. Encode current observation → latent s_t
  2. Predict next latent: s_pred = Predictor(s_t, a_t)
  3. Target: s_target = TargetEncoder(next_obs) [EMA, no grad]
  4. Prediction loss: ||s_pred - s_target||²
  5. SIGReg: enforce N(0,I) distribution on embeddings (prevents collapse)
  6. Update target encoder via EMA

Only 1 hyperparameter: lambda (SIGReg weight). Trains stably end-to-end.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA
from data.dataset import GameDataset


def train_jepa(
    csv_file="data/sciencebirds_data.csv",
    batch_size=1024,
    epochs=28,
    lr=1e-3,
    sigreg_lambda=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints"
):
    print(f"=== Vortaz Labs — LeWorldModel (LeWM) Training on {device} ===")
    print(f"Loss: L_prediction + SIGReg (lambda={sigreg_lambda})\n")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run data/collector.py first.")
        return

    os.makedirs(save_dir, exist_ok=True)

    dataset = GameDataset(csv_file, fit_norm=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset):,} samples | "
          f"{len(dataloader)} batches/epoch\n")

    model = GameJEPA(
        obs_dim=164, action_dim=3, latent_dim=256,
        hidden_dim=512, ema_momentum=0.996,
        sigreg_lambda=sigreg_lambda,  # LeWM's only hyperparameter
        use_memory=False,       # disabled for pretraining (no sequences)
        use_configurator=False,  # no context during pretraining
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {trainable:,} trainable parameters\n")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cos_sim = 0.0

        for batch_idx, (x, a, y, score) in enumerate(dataloader):
            x, a, y = x.to(device), a.to(device), y.to(device)

            optimizer.zero_grad()

            # LeWM loss: prediction + SIGReg (just 2 terms)
            loss, s_pred, s_target, _ = model(x, a, y, loss_type="lewm")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update — critical for JEPA
            model.update_target_encoder()

            total_loss += loss.item()

            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    s_pred, s_target, dim=-1
                ).mean().item()
                total_cos_sim += cos_sim

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (batch_idx + 1)
                avg_cos = total_cos_sim / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| CosSim: {avg_cos:.4f} "
                      f"| {elapsed:.0f}s")

        avg_loss = total_loss / len(dataloader)
        avg_cos = total_cos_sim / len(dataloader)
        print(f"\n==> Epoch {epoch+1} | Loss: {avg_loss:.4f} "
              f"| CosSim: {avg_cos:.4f}\n")

        ckpt_path = os.path.join(save_dir, f"game_jepa_ep{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'cosine_similarity': avg_cos,
        }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.1f} minutes.")
    print(f"Checkpoints: {save_dir}/game_jepa_ep1.pth .. ep{epochs}.pth")
    print(f"Recommended for decoder/critic training: ep{max(1, epochs-2)}.pth")


if __name__ == "__main__":
    train_jepa(batch_size=1024, epochs=28)
