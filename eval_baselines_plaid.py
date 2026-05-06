"""
Train and evaluate Plain MLP Dynamics and C-SWM baselines on PLAID AirfRANS.

Same SymbolicEncoder + CausalPredictor architecture as SY-JEPA's PLAID model;
only the training objective differs:
  Plain MLP  — supervised MSE(z_pred, z_next), both encoder calls in graph
  C-SWM      — contrastive hinge loss, cyclic batch negatives (Kipf et al. 2020)
  SY-JEPA    — JEPA prediction loss + SIGReg + EMA target encoder (reference)

After training, a decoder is fitted on the frozen baseline encoder/predictor
and evaluated on a held-out test set (80/20 split, seed 42).

SY-JEPA reference (from checkpoints/plaid_symbolic_jepa.pth):
  Decoded MAE:   0.6240
  Latent CosSim: 0.5210

Run:
    python eval_baselines_plaid.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plaid.dataset import PLAIDDataset, POOLED_DIM, ACTION_DIM
from plaid.symbolic_jepa import SymbolicDecoder

# SY-JEPA reference scores (from evaluate_plaid.py on trained checkpoint)
SJEPA_MAE    = 0.6240
SJEPA_COSINE = 0.5210


# ── Baseline model (same arch as SymbolicJEPA, no EMA / SIGReg) ──────────────

class PlaidBaseline(nn.Module):
    """
    SymbolicEncoder + CausalPredictor with identical architecture to
    SymbolicJEPA.  No EMA target encoder, no SIGReg — only the training
    objective changes (MSE or C-SWM contrastive).
    """

    def __init__(self, input_dim: int = POOLED_DIM,
                 action_dim: int = ACTION_DIM,
                 latent_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([z, a], dim=-1))

    def forward(self, x, a, y):
        z_pred = self.predict(self.encode(x), a)
        z_next = self.encode(y)          # no stop-gradient
        return z_pred, z_next


# ── Loss functions ────────────────────────────────────────────────────────────

def _mse_loss(z_pred, z_next):
    return F.mse_loss(z_pred, z_next)


def _cswm_loss(z_pred, z_next, margin: float = 1.0, n_neg: int = 5):
    """Kipf et al. ICLR 2020: L2 hinge contrastive with cyclic-roll negatives."""
    d_pos  = (z_pred - z_next).pow(2).sum(dim=-1)
    total  = torch.zeros(1, device=z_pred.device)
    for k in range(1, n_neg + 1):
        z_neg = torch.roll(z_next, k, dims=0)
        d_neg = (z_pred - z_neg).pow(2).sum(dim=-1)
        total = total + torch.clamp(margin + d_pos - d_neg, min=0.0).mean()
    return total / n_neg


# ── Training helper ───────────────────────────────────────────────────────────

def _train_and_eval(name: str, train_loader, test_loader,
                    device: str, epochs: int = 30, lr: float = 1e-3) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    use_cswm = "c-swm" in name.lower() or "cswm" in name.lower()

    model     = PlaidBaseline().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, a, y, _ in train_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            optimizer.zero_grad()
            z_pred, z_next = model(x, a, y)
            loss = _cswm_loss(z_pred, z_next) if use_cswm else _mse_loss(z_pred, z_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:>3}/{epochs} | Loss: {avg:.6f} | "
                  f"{time.time()-t0:.0f}s")

    # ── Decoder phase (15 epochs, baseline frozen) ────────────────────────────
    print(f"  Training decoder (15 epochs)...")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    decoder   = SymbolicDecoder(latent_dim=256, output_dim=POOLED_DIM).to(device)
    dec_opt   = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(15):
        decoder.train()
        for x, a, y, _ in train_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            dec_opt.zero_grad()
            with torch.no_grad():
                z_pred = model.predict(model.encode(x), a)
            loss = F.mse_loss(decoder(z_pred), y)
            loss.backward()
            dec_opt.step()

    # ── Evaluation on test set ────────────────────────────────────────────────
    model.eval()
    decoder.eval()
    all_mae, all_cos = [], []

    with torch.no_grad():
        for x, a, y, _ in test_loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            z_t    = model.encode(x)
            z_pred = model.predict(z_t, a)
            z_next = model.encode(y)              # online encoder as reference

            decoded = decoder(z_pred)
            all_mae.append((decoded - y).abs().mean().item())
            all_cos.append(
                F.cosine_similarity(z_pred, z_next, dim=-1).mean().item()
            )

    avg_mae = float(np.mean(all_mae))
    avg_cos = float(np.mean(all_cos))
    print(f"  Decoded MAE:   {avg_mae:.4f}  (SY-JEPA: {SJEPA_MAE:.4f})")
    print(f"  Latent CosSim: {avg_cos:.4f}  (SY-JEPA: {SJEPA_COSINE:.4f})")
    return {"mae": avg_mae, "cosine": avg_cos}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PLAID AirfRANS — SY-JEPA vs World Model Baselines")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    data_dir = str(Path(__file__).resolve().parent / "plaid" / "data")
    print(f"\n  Loading dataset from {data_dir}...")
    try:
        dataset = PLAIDDataset(data_dir, max_pairs=5000)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run: python plaid/download_data.py first")
        return

    train_n = int(0.8 * len(dataset))
    test_n  = len(dataset) - train_n
    train_ds, test_ds = random_split(
        dataset, [train_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {train_n} | Test: {test_n}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)

    results = {
        "SY-JEPA": {"mae": SJEPA_MAE, "cosine": SJEPA_COSINE,
                    "note": "from checkpoints/plaid_symbolic_jepa.pth"},
    }

    for name in ["Plain MLP Dynamics", "C-SWM"]:
        results[name] = _train_and_eval(
            name, train_loader, test_loader, device
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS (PLAID AirfRANS — lower MAE / higher CosSim = better)")
    print("=" * 60)
    print(f"  {'Model':<24} {'Decoded MAE':>12} {'Latent CosSim':>14}")
    print(f"  {'-'*52}")
    for mname, r in results.items():
        marker = " ←" if mname == "SY-JEPA" else ""
        print(f"  {mname:<24} {r['mae']:>12.4f} {r['cosine']:>14.4f}{marker}")
    print()

    out = Path(__file__).resolve().parent / "plaid_baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out}")


if __name__ == "__main__":
    main()
