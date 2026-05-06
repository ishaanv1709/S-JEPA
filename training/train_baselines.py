"""
Vortaz Labs — Baseline World Model Training
============================================
Trains two baselines on the same 50K Science Birds trajectories as SY-JEPA.
Both reuse the identical Encoder + Predictor architecture for a fair comparison.
Only the training objective differs.

Baseline A — Plain MLP Dynamics:
  Loss = MSE(predictor(encoder(s_t), a_t), encoder(s_{t+1}))
  No EMA, no SIGReg, no stop-gradient. Pure supervised dynamics.

Baseline B — C-SWM (Kipf et al., ICLR 2020):
  Contrastive loss pushes predicted z_{t+1} toward the true next embedding
  and away from randomly sampled negatives — no EMA, no SIGReg.

After training, run train_baseline_critics.py to fit critics on the
baseline latent spaces. Then use run_ablations_live.py --baseline to
evaluate on Unity.

Usage:
    python -m training.train_baselines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA
from data.dataset import GameDataset


# ── Shared encoder/predictor (identical architecture to SY-JEPA) ──────────────
class BaselineModel(nn.Module):
    """
    Encoder + Predictor with the same architecture as SY-JEPA.
    No EMA copy, no SIGReg, no stop-gradient target.
    Used for both Plain MLP and C-SWM baselines.
    """
    def __init__(self, obs_dim=164, action_dim=3,
                 latent_dim=256, hidden_dim=512):
        super().__init__()
        # Encoder: same as SY-JEPA encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # Predictor: same as SY-JEPA predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def encode(self, s):
        return self.encoder(s)

    def predict(self, z, a):
        return self.predictor(torch.cat([z, a], dim=-1))

    def forward(self, s_t, a_t, s_next):
        """Returns (z_pred, z_next) — both with gradients."""
        z_t    = self.encode(s_t)
        z_pred = self.predict(z_t, a_t)
        z_next = self.encode(s_next)   # no stop-gradient
        return z_pred, z_next


# ── Option A: Plain MLP Dynamics ─────────────────────────────────────────────
def train_plain_mlp(
    csv_file="data/sciencebirds_data.csv",
    batch_size=1024,
    epochs=28,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):
    print(f"\n{'='*60}")
    print(f"  Baseline A: Plain MLP Dynamics (supervised MSE)")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found. Run data/collector.py first.")
        return

    os.makedirs(save_dir, exist_ok=True)

    dataset    = GameDataset(csv_file, fit_norm=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset):,} samples | {len(dataloader)} batches/epoch\n")

    model     = BaselineModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,}\n")

    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cos  = 0.0

        for batch_idx, (x, a, y, _) in enumerate(dataloader):
            x, a, y = x.to(device), a.to(device), y.to(device)
            optimizer.zero_grad()

            z_pred, z_next = model(x, a, y)
            loss = F.mse_loss(z_pred, z_next)   # plain supervised MSE

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_cos += F.cosine_similarity(z_pred, z_next, dim=-1).mean().item()

            if (batch_idx + 1) % 50 == 0:
                elapsed  = time.time() - start
                avg_loss = total_loss / (batch_idx + 1)
                avg_cos  = total_cos  / (batch_idx + 1)
                print(f"Ep[{epoch+1}/{epochs}] B[{batch_idx+1}] "
                      f"Loss:{avg_loss:.4f} CosSim:{avg_cos:.4f} {elapsed:.0f}s")

        avg_loss = total_loss / len(dataloader)
        avg_cos  = total_cos  / len(dataloader)
        print(f"\n==> Epoch {epoch+1} | Loss:{avg_loss:.4f} CosSim:{avg_cos:.4f}\n")

    ckpt = os.path.join(save_dir, "baseline_plain_mlp.pth")
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    print(f"Saved: {ckpt}  ({(time.time()-start)/60:.1f} min)")
    return model


# ── Option B: C-SWM ──────────────────────────────────────────────────────────
def cswm_loss(z_pred, z_next, margin=1.0, n_neg=5):
    """
    Contrastive loss from Kipf et al. (ICLR 2020).
    Pushes z_pred toward z_next (positive) and away from z_neg
    (other items in the batch, used as negatives).

    L = Σ max(0, margin - d(z_pred, z_neg) + d(z_pred, z_next))

    d = L2 distance (not cosine — C-SWM original uses L2).
    """
    B = z_pred.shape[0]
    d_pos = (z_pred - z_next).pow(2).sum(dim=-1)          # (B,)

    # Sample negatives by rolling the batch
    total_loss = torch.zeros(1, device=z_pred.device)
    for k in range(1, n_neg + 1):
        z_neg  = torch.roll(z_next, k, dims=0)            # cyclic shift
        d_neg  = (z_pred - z_neg).pow(2).sum(dim=-1)      # (B,)
        hinge  = torch.clamp(margin + d_pos - d_neg, min=0.0)
        total_loss = total_loss + hinge.mean()

    return total_loss / n_neg


def train_cswm(
    csv_file="data/sciencebirds_data.csv",
    batch_size=1024,
    epochs=28,
    lr=1e-3,
    margin=1.0,
    n_neg=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):
    print(f"\n{'='*60}")
    print(f"  Baseline B: C-SWM (Kipf et al., ICLR 2020)")
    print(f"  Contrastive margin={margin}, negatives={n_neg}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found. Run data/collector.py first.")
        return

    os.makedirs(save_dir, exist_ok=True)

    dataset    = GameDataset(csv_file, fit_norm=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset):,} samples | {len(dataloader)} batches/epoch\n")

    model     = BaselineModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,}\n")

    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cos  = 0.0

        for batch_idx, (x, a, y, _) in enumerate(dataloader):
            x, a, y = x.to(device), a.to(device), y.to(device)
            optimizer.zero_grad()

            z_pred, z_next = model(x, a, y)
            loss = cswm_loss(z_pred, z_next, margin=margin, n_neg=n_neg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_cos += F.cosine_similarity(z_pred, z_next, dim=-1).mean().item()

            if (batch_idx + 1) % 50 == 0:
                elapsed  = time.time() - start
                avg_loss = total_loss / (batch_idx + 1)
                avg_cos  = total_cos  / (batch_idx + 1)
                print(f"Ep[{epoch+1}/{epochs}] B[{batch_idx+1}] "
                      f"Loss:{avg_loss:.4f} CosSim:{avg_cos:.4f} {elapsed:.0f}s")

        avg_loss = total_loss / len(dataloader)
        avg_cos  = total_cos  / len(dataloader)
        print(f"\n==> Epoch {epoch+1} | Loss:{avg_loss:.4f} CosSim:{avg_cos:.4f}\n")

    ckpt = os.path.join(save_dir, "baseline_cswm.pth")
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    print(f"Saved: {ckpt}  ({(time.time()-start)/60:.1f} min)")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training Baseline A: Plain MLP Dynamics (~26 min)...")
    train_plain_mlp(device=device)

    print("\nTraining Baseline B: C-SWM (~26 min)...")
    train_cswm(device=device)

    print("\n" + "="*60)
    print("Both baselines trained.")
    print("Next step: train critics on baseline latent spaces:")
    print("  python -m training.train_baseline_critics")
    print("Then run Unity ablation:")
    print("  python run_ablations_live.py --level 1 --baselines")
    print("="*60)
