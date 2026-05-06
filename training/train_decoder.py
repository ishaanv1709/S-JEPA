"""
Train Decoder on PREDICTOR outputs (not target encoder outputs).
Critical lesson from Volt codebase: decoder must be trained on predictor
outputs to ensure it learns to interpret the world model's predictions.
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

from models.world_model import GameJEPA, GameDecoder
from data.dataset import GameDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
JEPA_CHECKPOINT = "checkpoints/game_jepa_ep6.pth"  # recommended
EPOCHS = 44


def train_decoder(jepa_ckpt=JEPA_CHECKPOINT, epochs=EPOCHS):
    print(f"=== Vortaz Labs — Decoder Training on {device} ===\n")

    if not os.path.exists(jepa_ckpt):
        print(f"Error: {jepa_ckpt} not found. Run train_jepa.py first.")
        return

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

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    # Decoder
    decoder = GameDecoder(latent_dim=256, output_dim=164).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.SmoothL1Loss()

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training decoder on PREDICTOR outputs")
    print(f"Train: {len(train_ds):,} | Test: {len(test_ds):,}\n")

    best_loss = float('inf')
    for epoch in range(epochs):
        decoder.train()
        total_loss, n = 0, 0
        for x, a, y, score in train_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)

            with torch.no_grad():
                s_pred, _ = jepa(x, a)  # predictor output, NOT target encoder

            y_hat = decoder(s_pred)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / n
        scheduler.step(avg)
        lr = optimizer.param_groups[0]['lr']
        tag = ""
        if avg < best_loss:
            best_loss = avg
            torch.save(decoder.state_dict(), "checkpoints/game_decoder.pth")
            tag = " *best*"
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg:.6f} "
              f"| LR: {lr:.6f}{tag}")

    # Evaluate
    print(f"\n--- Test evaluation ({test_size:,} samples) ---\n")
    decoder.load_state_dict(
        torch.load("checkpoints/game_decoder.pth", map_location=device)
    )
    decoder.eval()

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, a, y, score in test_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            s_pred, _ = jepa(x, a)
            pred = decoder(s_pred)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)
    errors = np.abs(preds - targets)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print(f"Overall MAE:  {mae:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")


if __name__ == "__main__":
    train_decoder()
