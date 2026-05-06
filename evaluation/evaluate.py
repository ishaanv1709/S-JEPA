"""
Evaluation — World model prediction accuracy metrics.

Tests how well the JEPA world model predicts shot outcomes
when decoded back to observation space.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA, GameDecoder
from models.critic import Critic
from data.dataset import GameDataset


def evaluate_world_model(jepa_ckpt="checkpoints/game_jepa_ep6.pth",
                         decoder_ckpt="checkpoints/game_decoder.pth",
                         critic_ckpt="checkpoints/game_critic.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Vortaz Labs — World Model Evaluation on {device} ===\n")

    # Load models
    jepa = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                    hidden_dim=512, use_memory=False,
                    use_configurator=False).to(device)
    if os.path.exists(jepa_ckpt):
        jepa.load_state_dict(
            torch.load(jepa_ckpt, map_location=device)['model_state_dict']
        )
    jepa.eval()

    decoder = GameDecoder(latent_dim=256, output_dim=164).to(device)
    if os.path.exists(decoder_ckpt):
        decoder.load_state_dict(
            torch.load(decoder_ckpt, map_location=device)
        )
    decoder.eval()

    critic = Critic(latent_dim=256, action_dim=3).to(device)
    if os.path.exists(critic_ckpt):
        critic.load_state_dict(
            torch.load(critic_ckpt, map_location=device)
        )
    critic.eval()

    # Dataset
    csv = "data/sciencebirds_data.csv"
    if not os.path.exists(csv):
        print(f"{csv} not found. Run data/collector.py first.")
        return

    ds = GameDataset(csv, fit_norm=False)
    total = len(ds)
    test_size = int(total * 0.1)
    _, test_ds = random_split(
        ds, [total - test_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    print(f"Evaluating on {test_size:,} test samples\n")

    # Collect predictions
    all_preds_norm, all_tgts_norm = [], []
    all_energies, all_scores = [], []

    with torch.no_grad():
        for x, a, y, score in test_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            score = score.to(device)

            s_current = jepa.encoder(x)
            s_pred, _ = jepa(x, a)
            pred = decoder(s_pred)
            energy = critic(s_current, s_pred, a).squeeze(-1)

            all_preds_norm.append(pred.cpu().numpy())
            all_tgts_norm.append(y.cpu().numpy())
            all_energies.append(energy.cpu().numpy())
            all_scores.append(score.cpu().numpy())

    preds = np.concatenate(all_preds_norm)
    targets = np.concatenate(all_tgts_norm)
    energies = np.concatenate(all_energies)
    scores = np.concatenate(all_scores)

    # Overall metrics
    errors = np.abs(preds - targets)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Overall MAE (normalized)':<30} {mae:>10.6f}")
    print(f"{'Overall RMSE (normalized)':<30} {rmse:>10.6f}")

    # Per-feature group analysis
    # Birds: 0-19, Blocks: 20-139, Pigs: 140-159, Slingshot: 160-161, Global: 162-163
    groups = {
        "Birds (0-19)": (0, 20),
        "Blocks (20-139)": (20, 140),
        "Pigs (140-159)": (140, 160),
        "Slingshot (160-161)": (160, 162),
        "Global (162-163)": (162, 164),
    }

    print(f"\n{'Feature Group':<30} {'MAE':>10} {'RMSE':>10}")
    print("-" * 52)
    for name, (start, end) in groups.items():
        group_mae = np.mean(np.abs(preds[:, start:end] - targets[:, start:end]))
        group_rmse = np.sqrt(np.mean((preds[:, start:end] - targets[:, start:end]) ** 2))
        print(f"{name:<30} {group_mae:>10.6f} {group_rmse:>10.6f}")

    # Critic evaluation
    try:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(-energies, scores)
        print(f"\n{'Critic rank correlation':<30} {corr:>10.4f}")
        print(f"{'p-value':<30} {pval:>10.2e}")
    except ImportError:
        pass

    # Cosine similarity in latent space
    all_cos = []
    with torch.no_grad():
        for x, a, y, score in test_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            s_pred, _ = jepa(x, a)
            s_target = jepa.target_encoder(y)
            cos = torch.nn.functional.cosine_similarity(
                s_pred, s_target, dim=-1
            ).mean().item()
            all_cos.append(cos)

    avg_cos = np.mean(all_cos)
    print(f"\n{'Latent cosine similarity':<30} {avg_cos:>10.4f}")
    print(f"{'(1.0 = perfect prediction)':<30}")

    # Sample predictions
    print(f"\n--- Sample predictions (first 3) ---")
    for i in range(min(3, len(preds))):
        print(f"\n  Sample {i+1}:")
        print(f"  Predicted: {preds[i, :5]}...")
        print(f"  Target:    {targets[i, :5]}...")
        print(f"  Error:     {errors[i, :5]}...")


if __name__ == "__main__":
    evaluate_world_model()
