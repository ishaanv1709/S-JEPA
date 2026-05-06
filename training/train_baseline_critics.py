"""
Train Energy Critics on top of the two baseline world models.
Same critic architecture and training procedure as SY-JEPA — only the
encoder/predictor weights differ.

Requires:
  checkpoints/baseline_plain_mlp.pth   (from train_baselines.py)
  checkpoints/baseline_cswm.pth        (from train_baselines.py)

Outputs:
  checkpoints/baseline_plain_critic.pth
  checkpoints/baseline_cswm_critic.pth

Usage:
    python -m training.train_baseline_critics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_baselines import BaselineModel
from models.critic import Critic
from data.dataset import GameDataset


def train_critic_on_baseline(
    baseline_ckpt: str,
    critic_save:   str,
    csv_file="data/sciencebirds_data.csv",
    batch_size=1024,
    epochs=15,
    lr=1e-3,
    margin=1.0,
    k_neg=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\nTraining critic on: {baseline_ckpt}")

    # Load frozen baseline encoder + predictor
    baseline = BaselineModel().to(device)
    state = torch.load(baseline_ckpt, map_location=device)
    baseline.load_state_dict(state["model_state_dict"])
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    # Fresh critic (same architecture as SY-JEPA critic)
    critic    = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)

    dataset    = GameDataset(csv_file, fit_norm=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    for epoch in range(epochs):
        critic.train()
        total_loss = 0.0

        for x, a, y, _ in dataloader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                z_t    = baseline.encode(x)
                z_pred = baseline.predict(z_t, a)  # positive predicted next state

            # Negatives: roll batch to get "wrong" next states
            loss = torch.tensor(0.0, device=device)
            e_pos = critic(z_t, z_pred, a)
            for k in range(1, k_neg + 1):
                z_neg = torch.roll(z_pred, k, dims=0)
                e_neg = critic(z_t, z_neg, a)
                loss  = loss + torch.clamp(e_pos - e_neg + margin, min=0.0).mean()
            loss = loss / k_neg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")

    os.makedirs(os.path.dirname(critic_save) or ".", exist_ok=True)
    torch.save(critic.state_dict(), critic_save)
    print(f"  Saved critic: {critic_save}\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, ckpt, critic_out in [
        ("Plain MLP", "checkpoints/baseline_plain_mlp.pth",
                      "checkpoints/baseline_plain_critic.pth"),
        ("C-SWM",     "checkpoints/baseline_cswm.pth",
                      "checkpoints/baseline_cswm_critic.pth"),
    ]:
        if not Path(ckpt).exists():
            print(f"SKIP {name}: {ckpt} not found. Run train_baselines.py first.")
            continue
        train_critic_on_baseline(
            baseline_ckpt=ckpt,
            critic_save=critic_out,
            device=device,
        )

    print("Done. Next:")
    print("  python run_ablations_live.py --level 1 --baselines")
