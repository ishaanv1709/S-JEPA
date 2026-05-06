"""
Vortaz Labs — PLAID Symbolic-JEPA Evaluation
===============================================
Comprehensive evaluation for the LossFunk proposal.

Tests:
  1. Prediction accuracy: latent cosine sim, decoded field error
  2. In-context learning: adapt to new Reynolds numbers without retraining
  3. Zero-shot shape transfer: train on NACA 4-digit, test on NACA 5-digit
  4. Critic quality: Spearman ρ against ground truth
  5. Multi-step rollout stability

Usage: python plaid/evaluate_plaid.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaid.symbolic_jepa import SymbolicJEPA, SymbolicDecoder
from plaid.dataset import PLAIDDataset, POOLED_DIM, ACTION_DIM


def test_prediction_accuracy(model, decoder, test_loader, device):
    """Test 1: Basic prediction accuracy."""
    print(f"\n  Test 1: Prediction Accuracy")
    print(f"  {'─'*40}")

    all_mae = []
    all_cos = []

    model.eval()
    decoder.eval()

    with torch.no_grad():
        for x, a, y, _ in tqdm(test_loader, desc="    Predicting", unit="batch",
                                bar_format='{l_bar}{bar:20}{r_bar}'):
            x, a, y = x.to(device), a.to(device), y.to(device)

            s_current = model.encoder(x)
            s_pred = model.predictor(s_current, a)
            s_target = model.target_encoder(y)

            decoded = decoder(s_pred)
            mae = (decoded - y).abs().mean().item()
            cos = F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()

            all_mae.append(mae)
            all_cos.append(cos)

    avg_mae = np.mean(all_mae)
    avg_cos = np.mean(all_cos)

    print(f"    Decoded MAE:   {avg_mae:.6f}")
    print(f"    Latent CosSim: {avg_cos:.4f}")

    return {"mae": avg_mae, "cosine_sim": avg_cos}


def test_zero_shot_transfer(model, decoder, dataset, device):
    """Test 3: Zero-shot shape transfer (NACA 4→5 digit)."""
    print(f"\n  Test 3: Zero-Shot Shape Transfer (NACA 4-digit → 5-digit)")
    print(f"  {'─'*40}")

    naca4_idx = dataset.get_naca4_indices()
    naca5_idx = dataset.get_naca5_indices()

    print(f"    NACA 4-digit sims: {len(naca4_idx)}")
    print(f"    NACA 5-digit sims: {len(naca5_idx)}")

    if len(naca5_idx) == 0:
        print(f"    No NACA 5-digit sims found. Skipping.")
        return {"naca4_to_5_mae": None}

    # Create test pairs: NACA4 → NACA5
    rng = np.random.RandomState(42)
    test_pairs = []
    for _ in range(min(500, len(naca4_idx) * len(naca5_idx))):
        i = rng.choice(naca4_idx)
        j = rng.choice(naca5_idx)
        test_pairs.append((i, j))

    model.eval()
    decoder.eval()

    maes = []
    cos_sims = []

    with torch.no_grad():
        for i, j in tqdm(test_pairs, desc="    Cross-shape", unit="pair",
                          bar_format='{l_bar}{bar:20}{r_bar}'):
            x = torch.tensor(dataset.features[i]).unsqueeze(0).to(device)
            y = torch.tensor(dataset.features[j]).unsqueeze(0).to(device)

            delta_aoa = (dataset.aoas[j] - dataset.aoas[i]) / max(dataset.aoa_std, 1e-6)
            delta_re = (dataset.reynolds[j] - dataset.reynolds[i]) / max(dataset.re_std, 1e-6)
            a = torch.tensor([[delta_aoa, delta_re]], dtype=torch.float32).to(device)

            s_current = model.encoder(x)
            s_pred = model.predictor(s_current, a)
            s_target = model.target_encoder(y)

            decoded = decoder(s_pred)
            mae = (decoded - y).abs().mean().item()
            cos = F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()

            maes.append(mae)
            cos_sims.append(cos)

    avg_mae = np.mean(maes)
    avg_cos = np.mean(cos_sims)

    print(f"    Cross-shape MAE:   {avg_mae:.6f}")
    print(f"    Cross-shape CosSim: {avg_cos:.4f}")

    return {"naca4_to_5_mae": avg_mae, "naca4_to_5_cos": avg_cos}


def test_in_context_reynolds(model, decoder, dataset, device):
    """Test 2: In-context adaptation to different Reynolds numbers."""
    print(f"\n  Test 2: In-Context Reynolds Adaptation")
    print(f"  {'─'*40}")

    # Split by Reynolds number
    re_values = dataset.reynolds
    re_low = re_values < np.percentile(re_values, 33)
    re_mid = (re_values >= np.percentile(re_values, 33)) & (re_values < np.percentile(re_values, 67))
    re_high = re_values >= np.percentile(re_values, 67)

    regimes = {
        "Low Re (<2.7M)": re_low,
        "Mid Re (2.7-4.7M)": re_mid,
        "High Re (>4.7M)": re_high,
    }

    model.eval()
    decoder.eval()

    results = {}
    rng = np.random.RandomState(42)

    for regime_name, mask in regimes.items():
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue

        maes = []
        with torch.no_grad():
            for _ in tqdm(range(min(200, len(indices) * (len(indices) - 1))),
                          desc=f"    {regime_name:>20}", unit="pair", leave=False,
                          bar_format='{l_bar}{bar:15}{r_bar}'):
                i = rng.choice(indices)
                j = rng.choice(indices)
                if i == j:
                    continue

                x = torch.tensor(dataset.features[i]).unsqueeze(0).to(device)
                y = torch.tensor(dataset.features[j]).unsqueeze(0).to(device)

                delta_aoa = (dataset.aoas[j] - dataset.aoas[i]) / max(dataset.aoa_std, 1e-6)
                delta_re = (dataset.reynolds[j] - dataset.reynolds[i]) / max(dataset.re_std, 1e-6)
                a = torch.tensor([[delta_aoa, delta_re]], dtype=torch.float32).to(device)

                s_current = model.encoder(x)
                s_pred = model.predictor(s_current, a)
                decoded = decoder(s_pred)
                mae = (decoded - y).abs().mean().item()
                maes.append(mae)

        avg_mae = np.mean(maes) if maes else 0.0
        results[regime_name] = avg_mae
        print(f"    {regime_name}: MAE = {avg_mae:.6f} ({len(indices)} sims)")

    return results


def test_critic_quality(model, test_loader, device):
    """Test 4: Critic quality (Spearman ρ)."""
    print(f"\n  Test 4: Critic Quality (Physical Verifier)")
    print(f"  {'─'*40}")

    model.eval()
    all_energies = []
    all_scores = []

    with torch.no_grad():
        for x, a, y, score in tqdm(test_loader, desc="    Critic eval", unit="batch",
                                    bar_format='{l_bar}{bar:20}{r_bar}'):
            x, a = x.to(device), a.to(device)

            s_current = model.encoder(x)
            s_pred = model.predictor(s_current, a)
            energy = model.critic(s_current, s_pred, a).squeeze(-1)

            all_energies.append(energy.cpu().numpy())
            all_scores.append(score.numpy())

    energies = np.concatenate(all_energies)
    scores = np.concatenate(all_scores)

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(-energies, scores)
    except ImportError:
        rho, pval = 0.0, 1.0

    energy_range = energies.max() - energies.min()

    print(f"    Spearman ρ:    {rho:.4f} (target > 0.71)")
    print(f"    p-value:       {pval:.2e}")
    print(f"    Energy range:  {energy_range:.4f}")

    return {"spearman_rho": float(rho), "p_value": float(pval),
            "energy_range": float(energy_range)}


def test_multistep_stability(model, test_loader, device, n_steps=10):
    """Test 5: Multi-step rollout stability."""
    print(f"\n  Test 5: Multi-Step Rollout Stability ({n_steps} steps)")
    print(f"  {'─'*40}")

    model.eval()

    with torch.no_grad():
        x, a, y, _ = next(iter(test_loader))
        x, a = x.to(device), a.to(device)

        # Repeat same action for multi-step
        actions = [a for _ in range(n_steps)]
        rollout = model.multi_step_predict(x, actions)

        norms = [s.norm(dim=-1).mean().item() for s in rollout]
        print(f"    Step  | Latent Norm")
        print(f"    {'─'*25}")
        for i, norm in enumerate(norms):
            marker = " ← start" if i == 0 else (" ← end" if i == len(norms)-1 else "")
            print(f"    {i:>5} | {norm:>10.4f}{marker}")

        drift = abs(norms[-1] - norms[0]) / max(norms[0], 1e-6)
        print(f"    Drift: {drift:.4f} (< 0.5 = stable)")

    return {"norms": norms, "drift": drift}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — PLAID Symbolic-JEPA Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")

    # Load model
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_path = ckpt_dir / "plaid_symbolic_jepa.pth"

    if not ckpt_path.exists():
        print(f"\n  ERROR: No checkpoint at {ckpt_path}")
        print(f"  Run: python plaid/train_plaid.py first")
        sys.exit(1)

    print(f"\n  Loading model from {ckpt_path.name}...")
    model = SymbolicJEPA(input_dim=POOLED_DIM, action_dim=ACTION_DIM,
                         latent_dim=256, hidden_dim=256).to(device)
    decoder = SymbolicDecoder(latent_dim=256, output_dim=POOLED_DIM).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    print(f"  Model loaded (JEPA loss: {ckpt['jepa_loss']:.4f})")

    # Load dataset
    data_dir = str(Path(__file__).resolve().parent / "data")
    dataset = PLAIDDataset(data_dir, max_pairs=2000)

    # Use 20% as test
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    _, test_ds = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    print(f"  Test samples: {len(test_ds)}")

    # Run all tests
    results = {}

    results["prediction"] = test_prediction_accuracy(model, decoder, test_loader, device)
    results["reynolds"] = test_in_context_reynolds(model, decoder, dataset, device)
    results["transfer"] = test_zero_shot_transfer(model, decoder, dataset, device)
    results["critic"] = test_critic_quality(model, test_loader, device)
    results["stability"] = test_multistep_stability(model, test_loader, device)

    # Summary
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"  {'='*60}")
    print(f"  Prediction MAE:     {results['prediction']['mae']:.6f}")
    print(f"  Latent CosSim:      {results['prediction']['cosine_sim']:.4f}")
    print(f"  Critic Spearman ρ:  {results['critic']['spearman_rho']:.4f}")
    print(f"  Rollout drift:      {results['stability']['drift']:.4f}")
    if results['transfer'].get('naca4_to_5_mae') is not None:
        print(f"  Zero-shot NACA4→5:  {results['transfer']['naca4_to_5_mae']:.6f}")
    print(f"  Total time:         {total_time:.1f}s")

    # Save
    results_path = Path(__file__).resolve().parent / "evaluation_results.json"
    # Convert numpy types
    def clean(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
