"""
Vortaz Labs — PLAID Symbolic-JEPA Training
============================================
Trains the Symbolic-JEPA on AirfRANS data (100 simulations).

Training pipeline:
  Phase 1: Train Encoder + Predictor (JEPA loss + SIGReg)     ~10 epochs
  Phase 2: Train Critic (contrastive margin loss)              ~10 epochs
  Phase 3: Train Decoder (MSE reconstruction for eval)         ~10 epochs

Target: Complete training in <5 minutes on RTX 3050.

Usage: python plaid/train_plaid.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaid.symbolic_jepa import SymbolicJEPA, SymbolicDecoder
from plaid.dataset import PLAIDDataset, POOLED_DIM, ACTION_DIM


def train_jepa_phase(model, dataloader, epochs, lr, device):
    """Phase 1: Train encoder + predictor with LeWM loss."""
    print(f"\n  Phase 1: JEPA Training (Encoder + Predictor)")
    print(f"  Loss: L_prediction + SIGReg")
    print(f"  {'─'*40}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    epoch_bar = tqdm(range(epochs), desc="  Phase1 JEPA", unit="ep",
                     bar_format='{l_bar}{bar:20}{r_bar}')
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total_cos = 0.0

        for batch_idx, (x, a, y, score) in enumerate(dataloader):
            x, a, y = x.to(device), a.to(device), y.to(device)

            optimizer.zero_grad()
            loss, s_pred, s_target = model(x, a, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()

            total_loss += loss.item()
            with torch.no_grad():
                cos = F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()
                total_cos += cos

        avg_loss = total_loss / len(dataloader)
        avg_cos = total_cos / len(dataloader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", cos=f"{avg_cos:.4f}")

    return avg_loss, avg_cos


def train_critic_phase(model, dataloader, epochs, lr, device):
    """Phase 2: Train critic with contrastive margin loss."""
    print(f"\n  Phase 2: Critic Training (Physical Verifier)")
    print(f"  Loss: Contrastive margin (2.3x oversampling)")
    print(f"  {'─'*40}")

    # Freeze encoder + predictor, train only critic
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.critic.parameters(), lr=lr, weight_decay=1e-4)

    epoch_bar = tqdm(range(epochs), desc="  Phase2 Critic", unit="ep",
                     bar_format='{l_bar}{bar:20}{r_bar}')
    for epoch in epoch_bar:
        model.eval()  # encoder/predictor frozen
        model.critic.train()
        total_loss = 0.0

        for x, a, y, score in dataloader:
            x, a, y, score = x.to(device), a.to(device), y.to(device), score.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                s_current = model.encoder(x)
                s_pred = model.predictor(s_current, a)

            critic_loss = model.compute_critic_loss(
                s_current, s_pred, a, score,
                margin=1.0, oversample_ratio=2.3
            )
            critic_loss.backward()
            optimizer.step()
            total_loss += critic_loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

    # Unfreeze
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.predictor.parameters():
        p.requires_grad = True

    return avg_loss


def train_decoder_phase(model, decoder, dataloader, epochs, lr, device):
    """Phase 3: Train decoder for evaluation/interpretability."""
    print(f"\n  Phase 3: Decoder Training (for evaluation)")
    print(f"  Loss: MSE reconstruction")
    print(f"  {'─'*40}")

    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    epoch_bar = tqdm(range(epochs), desc="  Phase3 Decoder", unit="ep",
                     bar_format='{l_bar}{bar:20}{r_bar}')
    for epoch in epoch_bar:
        decoder.train()
        model.eval()
        total_loss = 0.0

        for x, a, y, score in dataloader:
            x, a, y = x.to(device), a.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                s_current = model.encoder(x)
                s_pred = model.predictor(s_current, a)

            decoded = decoder(s_pred)
            loss = F.mse_loss(decoded, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.6f}")

    return avg_loss


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — PLAID Symbolic-JEPA Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")

    # Check for data
    data_dir = str(Path(__file__).resolve().parent / "data")
    if not Path(data_dir).exists() or not (Path(data_dir) / "airfrans_processed.npz").exists():
        print(f"\n  No data found. Generating synthetic AirfRANS data...")
        from plaid.download_data import generate_synthetic_airfrans
        generate_synthetic_airfrans(data_dir, n_sims=1000)

    # Load dataset
    print(f"\n  Loading PLAID dataset...")
    dataset = PLAIDDataset(data_dir, max_pairs=5000)

    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"  Batches: {len(train_loader)}")

    # Create model
    model = SymbolicJEPA(
        input_dim=POOLED_DIM,
        action_dim=ACTION_DIM,
        latent_dim=256,
        hidden_dim=256,
        sigreg_lambda=1.0,
    ).to(device)

    decoder = SymbolicDecoder(latent_dim=256, output_dim=POOLED_DIM).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Model params: {total_params:,}")
    print(f"  Decoder params: {dec_params:,}")

    # Save dir
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Phase 1: JEPA
    t1 = time.time()
    jepa_loss, jepa_cos = train_jepa_phase(model, train_loader, epochs=15, lr=1e-3, device=device)
    time_phase1 = time.time() - t1

    # Phase 2: Critic
    t2 = time.time()
    critic_loss = train_critic_phase(model, train_loader, epochs=10, lr=5e-4, device=device)
    time_phase2 = time.time() - t2

    # Phase 3: Decoder
    t3 = time.time()
    dec_loss = train_decoder_phase(model, decoder, train_loader, epochs=10, lr=1e-3, device=device)
    time_phase3 = time.time() - t3

    # Save
    model_path = ckpt_dir / "plaid_symbolic_jepa.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'jepa_loss': jepa_loss,
        'critic_loss': critic_loss,
        'decoder_loss': dec_loss,
    }, str(model_path))
    print(f"\n  Saved to {model_path.name}")

    # Validation
    print(f"\n  {'='*50}")
    print(f"  VALIDATION")
    print(f"  {'='*50}")

    model.eval()
    decoder.eval()
    val_mae = 0.0
    val_cos = 0.0
    val_energies = []
    val_scores = []

    with torch.no_grad():
        for x, a, y, score in val_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)

            s_current = model.encoder(x)
            s_pred = model.predictor(s_current, a)
            s_target = model.target_encoder(y)

            decoded = decoder(s_pred)
            val_mae += (decoded - y).abs().mean().item()
            val_cos += F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()

            energy = model.critic(s_current, s_pred, a).squeeze(-1)
            val_energies.append(energy.cpu().numpy())
            val_scores.append(score.numpy())

    val_mae /= len(val_loader)
    val_cos /= len(val_loader)

    # Critic quality
    energies = np.concatenate(val_energies)
    scores = np.concatenate(val_scores)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(-energies, scores)
    except ImportError:
        rho, pval = 0.0, 1.0

    print(f"  Decoded MAE:     {val_mae:.6f}")
    print(f"  Latent CosSim:   {val_cos:.4f}")
    print(f"  Critic Spearman: {rho:.4f} (p={pval:.2e})")

    # Summary
    total_time = time.time() - t0
    print(f"\n  {'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  {'='*50}")
    print(f"  Phase 1 (JEPA):    {time_phase1:.1f}s | Loss: {jepa_loss:.4f} | CosSim: {jepa_cos:.4f}")
    print(f"  Phase 2 (Critic):  {time_phase2:.1f}s | Loss: {critic_loss:.4f}")
    print(f"  Phase 3 (Decoder): {time_phase3:.1f}s | Loss: {dec_loss:.6f}")
    print(f"  Total:             {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Spearman ρ:        {rho:.4f} (target > 0.71)")


if __name__ == "__main__":
    main()
