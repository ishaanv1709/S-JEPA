"""
Train Critic (Cost Module) — learns energy function from data.

The critic predicts -score_delta (negative score = energy to minimize)
from (current_latent, predicted_latent, action) triples.

Fixed from v1:
  - Critic now sees BOTH current and predicted latent (not just predicted)
  - Uses Huber loss for robustness to score outliers
  - Longer training with warmup
  - Gradient clipping for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA
from models.critic import Critic
from data.dataset import GameDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60


def find_best_jepa_checkpoint():
    """Find the best JEPA checkpoint in checkpoints/ directory."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("game_jepa_ep*.pth"))
    if not checkpoints:
        return None
    # Use the one recommended by training (usually 2/3 through)
    best_idx = max(0, len(checkpoints) * 2 // 3 - 1)
    return str(checkpoints[best_idx])


def train_critic(jepa_ckpt=None, epochs=EPOCHS):
    if jepa_ckpt is None:
        jepa_ckpt = find_best_jepa_checkpoint()
    if jepa_ckpt is None or not os.path.exists(jepa_ckpt):
        print("Error: No JEPA checkpoint found. Run train_jepa.py first.")
        return

    print(f"=== Vortaz Labs — Critic Training (v2) on {device} ===")
    print(f"JEPA checkpoint: {jepa_ckpt}\n")

    # Load frozen JEPA
    jepa = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                    hidden_dim=512, use_memory=False,
                    use_configurator=False).to(device)
    jepa.load_state_dict(
        torch.load(jepa_ckpt, map_location=device)['model_state_dict']
    )
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    # Dataset
    ds = GameDataset("data/sciencebirds_data.csv", fit_norm=False)
    total = len(ds)
    test_size = int(total * 0.1)
    train_ds, test_ds = random_split(
        ds, [total - test_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Critic (v2: takes current + predicted latent)
    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    optimizer = optim.AdamW(critic.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)  # robust to score outliers

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Critic v2: (s_current, s_pred, delta, action) -> energy")
    print(f"Train: {len(train_ds):,} | Test: {len(test_ds):,}")
    print(f"Loss: Huber | Optimizer: AdamW | Schedule: Cosine\n")

    best_loss = float('inf')
    for epoch in range(epochs):
        critic.train()
        total_loss, n = 0, 0

        for x, a, y, score in train_loader:
            x, a, score = x.to(device), a.to(device), score.to(device)

            # Get current and predicted latent from frozen JEPA
            with torch.no_grad():
                s_current = jepa.encoder(x)
                s_pred, _ = jepa(x, a)

            # Critic predicts energy = -score (lower energy = better)
            energy_pred = critic(s_current, s_pred, a).squeeze(-1)
            energy_target = -score  # negative normalized score

            loss = criterion(energy_pred, energy_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg = total_loss / n
        lr = optimizer.param_groups[0]['lr']
        tag = ""
        if avg < best_loss:
            best_loss = avg
            torch.save(critic.state_dict(), "checkpoints/game_critic.pth")
            tag = " *best*"

        if (epoch + 1) % 5 == 0 or tag:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg:.6f} "
                  f"| LR: {lr:.6f}{tag}")

    # Evaluate
    print(f"\n--- Critic evaluation ---")
    critic.load_state_dict(
        torch.load("checkpoints/game_critic.pth", map_location=device)
    )
    critic.eval()

    all_energies, all_scores = [], []
    with torch.no_grad():
        for x, a, y, score in test_loader:
            x, a = x.to(device), a.to(device)
            s_current = jepa.encoder(x)
            s_pred, _ = jepa(x, a)
            energy = critic(s_current, s_pred, a).squeeze(-1)
            all_energies.append(energy.cpu().numpy())
            all_scores.append(score.cpu().numpy())

    energies = np.concatenate(all_energies)
    scores = np.concatenate(all_scores)

    from scipy import stats as scipy_stats
    corr, pval = scipy_stats.spearmanr(-energies, scores)
    print(f"Spearman correlation (energy ranking vs score): {corr:.4f} "
          f"(p={pval:.2e})")

    # Also check how well critic separates zero-score vs positive-score shots
    zero_mask = scores == scores.min()  # normalized zero-score
    pos_mask = ~zero_mask
    if pos_mask.any() and zero_mask.any():
        energy_zero = energies[zero_mask].mean()
        energy_pos = energies[pos_mask].mean()
        print(f"Mean energy (zero-score shots): {energy_zero:.4f}")
        print(f"Mean energy (positive-score shots): {energy_pos:.4f}")
        print(f"Separation: {energy_zero - energy_pos:.4f} "
              f"(positive = critic correctly ranks)")


if __name__ == "__main__":
    train_critic()
