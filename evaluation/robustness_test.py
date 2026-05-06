"""
Vortaz Labs — Sim-to-Real Robustness Testing
===============================================
Tests the world model's latent dynamics resilience to domain shifts
WITHOUT retraining. Compares JEPA degradation vs LLM degradation.

Perturbation types:
  1. State noise:    Gaussian noise on observations (sensor error)
  2. Gravity shift:  ±20% modification to physics constants
  3. Scale shift:    Different coordinate scaling
  4. Dropout noise:  Random features zeroed out (visual clutter)
  5. Action noise:   Perturbation on action execution

For each perturbation level (0, 0.05, 0.1, 0.2, 0.5, 1.0):
  → Measure prediction accuracy (MAE, cosine similarity)
  → Measure critic quality (Spearman ρ)

Usage: python evaluation/robustness_test.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA, GameDecoder
from models.critic import Critic
from data.dataset import GameDataset


PERTURBATION_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def apply_perturbation(states, noise_type, level, rng=None):
    """
    Apply a specific perturbation to a batch of states.

    Returns perturbed states (same shape).
    """
    if rng is None:
        rng = np.random

    if level == 0.0:
        return states

    if noise_type == "gaussian":
        # Additive Gaussian noise (simulates sensor error)
        noise = torch.randn_like(states) * level
        return states + noise

    elif noise_type == "gravity":
        # Modify position-related features (simulate different gravity)
        # Birds: y at indices 2,6,10,14,18
        # Blocks: y at indices 23,29,35,...
        # Pigs: y at indices 142,146,...
        perturbed = states.clone()
        y_indices = []
        for i in range(5):  # birds
            y_indices.append(i * 4 + 2)
        for i in range(20):  # blocks
            y_indices.append(20 + i * 6 + 3)
        for i in range(5):  # pigs
            y_indices.append(140 + i * 4 + 2)

        gravity_scale = 1.0 + level * 0.4 * (2 * torch.rand(1).item() - 1)  # ±20% at level=0.5
        for idx in y_indices:
            if idx < perturbed.shape[1]:
                perturbed[:, idx] *= gravity_scale
        return perturbed

    elif noise_type == "scale":
        # Coordinate system scaling (different resolution)
        scale = 1.0 + level * (2 * torch.rand(1).item() - 1)
        return states * scale

    elif noise_type == "dropout":
        # Random feature zeroing (visual clutter / occlusion)
        mask = torch.bernoulli(torch.ones_like(states) * (1 - level * 0.5))
        return states * mask

    elif noise_type == "action_noise":
        # This is applied to actions, not states
        noise = torch.randn_like(states) * level * 0.3
        return states + noise

    return states


def evaluate_with_perturbation(jepa, decoder, critic, test_loader,
                                noise_type, level, device):
    """Run evaluation on test data with specified perturbation."""
    all_mae = []
    all_cos = []
    all_energies = []
    all_scores = []

    with torch.no_grad():
        for x, a, y, score in test_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)

            # Apply perturbation
            if noise_type == "action_noise":
                a = apply_perturbation(a, noise_type, level)
            else:
                x = apply_perturbation(x, noise_type, level)

            # Forward pass
            s_current = jepa.encoder(x)
            s_pred, _ = jepa(x, a)
            s_target = jepa.target_encoder(y)

            # Decode predictions
            pred = decoder(s_pred)

            # Metrics
            mae = (pred - y).abs().mean().item()
            cos = F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()
            energy = critic(s_current, s_pred, a).squeeze(-1)

            all_mae.append(mae)
            all_cos.append(cos)
            all_energies.append(energy.cpu().numpy())
            all_scores.append(score.numpy())

    # Critic correlation
    energies = np.concatenate(all_energies)
    scores = np.concatenate(all_scores)
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(-energies, scores)
    except ImportError:
        rho = 0.0

    return {
        "mae": np.mean(all_mae),
        "cosine_sim": np.mean(all_cos),
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — Sim-to-Real Robustness Testing")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}\n")

    # Load models
    print("  Loading trained models...")
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"

    jepa = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                    hidden_dim=512, use_memory=False,
                    use_configurator=False).to(device)

    # Find best checkpoint
    for name in ["game_jepa_best_v2.pth", "game_jepa_ep28.pth",
                  "game_jepa_ep8.pth", "game_jepa_ep6.pth"]:
        ckpt = ckpt_dir / name
        if ckpt.exists():
            jepa.load_state_dict(
                torch.load(str(ckpt), map_location=device)['model_state_dict']
            )
            print(f"  JEPA: {name}")
            break
    else:
        ckpts = sorted(ckpt_dir.glob("game_jepa_ep*.pth"))
        if ckpts:
            jepa.load_state_dict(
                torch.load(str(ckpts[-1]), map_location=device)['model_state_dict']
            )
            print(f"  JEPA: {ckpts[-1].name}")
    jepa.eval()

    decoder = GameDecoder(latent_dim=256, output_dim=164).to(device)
    for name in ["game_decoder_v2.pth", "game_decoder.pth"]:
        dec_path = ckpt_dir / name
        if dec_path.exists():
            decoder.load_state_dict(torch.load(str(dec_path), map_location=device))
            print(f"  Decoder: {name}")
            break
    decoder.eval()

    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    for name in ["game_critic_v2.pth", "game_critic.pth"]:
        crit_path = ckpt_dir / name
        if crit_path.exists():
            critic.load_state_dict(torch.load(str(crit_path), map_location=device))
            print(f"  Critic: {name}")
            break
    critic.eval()

    # Load test data
    data_dir = Path(__file__).resolve().parent.parent / "data"
    csv = data_dir / "sciencebirds_data_v2.csv"
    if not csv.exists():
        csv = data_dir / "sciencebirds_data.csv"
    if not csv.exists():
        print(f"  ERROR: No data found at {data_dir}")
        sys.exit(1)

    print(f"\n  Loading test data from {csv.name}...")
    ds = GameDataset(str(csv), fit_norm=False)
    test_size = min(int(len(ds) * 0.1), 10000)
    _, test_ds = random_split(ds, [len(ds) - test_size, test_size],
                              generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    print(f"  Test samples: {test_size:,}")

    # Run all perturbation tests
    noise_types = ["gaussian", "gravity", "scale", "dropout", "action_noise"]
    all_results = {}

    for noise_type in tqdm(noise_types, desc="  Perturbation types", unit="type",
                            bar_format='{l_bar}{bar:20}{r_bar}'):
        print(f"\n  {'-'*50}")
        print(f"  Perturbation: {noise_type.upper()}")
        print(f"  {'-'*50}")
        print(f"  {'Level':>8} | {'MAE':>10} | {'CosSim':>10} | {'Spearman':>10}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        results = []
        for level in tqdm(PERTURBATION_LEVELS, desc=f"    {noise_type:>14}",
                          unit="lvl", leave=False,
                          bar_format='{l_bar}{bar:15}{r_bar}'):
            metrics = evaluate_with_perturbation(
                jepa, decoder, critic, test_loader,
                noise_type, level, device
            )
            results.append({
                "level": level,
                **metrics,
            })
            print(f"  {level:>8.2f} | {metrics['mae']:>10.6f} | "
                  f"{metrics['cosine_sim']:>10.4f} | "
                  f"{metrics['spearman_rho']:>10.4f}")

        all_results[noise_type] = results

    # Summary
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  ROBUSTNESS SUMMARY")
    print(f"  {'='*60}")

    # Compute degradation (ratio of perturbed to clean)
    for noise_type in noise_types:
        clean = all_results[noise_type][0]
        worst = all_results[noise_type][-1]

        mae_degrad = worst['mae'] / max(clean['mae'], 1e-8)
        cos_degrad = clean['cosine_sim'] / max(worst['cosine_sim'], 1e-8)

        print(f"\n  {noise_type.upper():<15}")
        print(f"    MAE degradation (clean->worst):    {mae_degrad:.2f}x")
        print(f"    CosSim degradation (clean->worst):  {cos_degrad:.2f}x")
        print(f"    Spearman clean: {clean['spearman_rho']:.4f} -> "
              f"worst: {worst['spearman_rho']:.4f}")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save results
    results_path = Path(__file__).resolve().parent / "robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
