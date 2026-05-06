"""
Vortaz Labs — Cross-Domain Transfer Training
==============================================
THE critical experiment: Does the JEPA Predictor generalize across domains?

Protocol:
  1. Generate grasping training data
  2. Train Model A: GraspJEPA FROM SCRATCH (full model)
  3. Train Model B: GraspJEPA TRANSFER (frozen Predictor from Science Birds,
                    only encoder trains)
  4. Compare convergence speed, final accuracy, and sample efficiency

If Model B converges faster or achieves similar accuracy with less data,
the Predictor has learned domain-agnostic causal structure.

Usage: python training/train_transfer.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domains.grasp_simulator import (generate_grasping_data, OBS_DIM as GRASP_OBS,
                                      ACTION_DIM as GRASP_ACTION)
from domains.grasp_encoder import GraspJEPA


class GraspDataset(Dataset):
    """PyTorch Dataset for grasping domain data."""

    def __init__(self, csv_file, fit_norm=True, norm_path=None):
        self.df = pd.read_csv(csv_file)

        self.states = self.df[[f"s_{i}" for i in range(GRASP_OBS)]].values.astype(np.float32)
        self.actions = self.df[[f"a_{i}" for i in range(GRASP_ACTION)]].values.astype(np.float32)
        self.next_states = self.df[[f"ns_{i}" for i in range(GRASP_OBS)]].values.astype(np.float32)
        self.scores = self.df["score_delta"].values.astype(np.float32)

        if fit_norm:
            all_s = np.concatenate([self.states, self.next_states])
            self.s_mean = all_s.mean(axis=0)
            self.s_std = all_s.std(axis=0)
            self.s_std[self.s_std < 1e-6] = 1.0
            self.a_mean = self.actions.mean(axis=0)
            self.a_std = self.actions.std(axis=0)
            self.a_std[self.a_std < 1e-6] = 1.0

            if norm_path:
                np.savez(norm_path, s_mean=self.s_mean, s_std=self.s_std,
                         a_mean=self.a_mean, a_std=self.a_std)
                print(f"  Norm stats saved to {norm_path}")

        self.states_n = (self.states - self.s_mean) / self.s_std
        self.actions_n = (self.actions - self.a_mean) / self.a_std
        self.next_states_n = (self.next_states - self.s_mean) / self.s_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (torch.tensor(self.states_n[idx]),
                torch.tensor(self.actions_n[idx]),
                torch.tensor(self.next_states_n[idx]),
                torch.tensor(self.scores[idx]))


def train_model(model, dataloader, epochs, lr, device, label="Model"):
    """Train a GraspJEPA model and return loss history."""
    # Differential learning rates: encoder/adapter at full LR,
    # unfrozen predictor layers at 10x lower LR
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    adapter_params = [p for p in model.adapter.parameters() if p.requires_grad] if hasattr(model, 'adapter') and not isinstance(model.adapter, torch.nn.Identity) else []
    predictor_params = [p for p in model.predictor.parameters() if p.requires_grad]

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": lr})
    if predictor_params:
        param_groups.append({"params": predictor_params, "lr": lr * 0.1})  # 10x lower

    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    history = []
    epoch_bar = tqdm(range(epochs), desc=f"  {label}", unit="ep",
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            total_loss += loss.item()
            with torch.no_grad():
                cos = F.cosine_similarity(s_pred, s_target, dim=-1).mean().item()
                total_cos += cos

        avg_loss = total_loss / len(dataloader)
        avg_cos = total_cos / len(dataloader)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "cos_sim": avg_cos})
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", cos=f"{avg_cos:.4f}")

    return history


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — Cross-Domain Transfer Experiment")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}\n")

    # Step 1: Generate grasping data (if not exists)
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    grasp_csv = data_dir / "grasping_data.csv"

    if not grasp_csv.exists():
        print("  Step 1: Generating grasping training data...")
        generate_grasping_data(n_episodes=1000, episode_length=50,
                               output_dir=str(data_dir))
    else:
        print(f"  Step 1: Using existing data ({grasp_csv.name})")

    # Step 2: Load dataset
    print(f"\n  Step 2: Loading grasping dataset...")
    norm_path = str(data_dir / "grasp_norm_stats.npz")
    dataset = GraspDataset(str(grasp_csv), fit_norm=True, norm_path=norm_path)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    print(f"  Samples: {len(dataset):,} | Batches: {len(dataloader)}")

    # Step 3: Find Science Birds checkpoint for transfer
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    sb_ckpt = None
    for name in ["game_jepa_best_v2.pth", "game_jepa_ep28.pth",
                  "game_jepa_ep8.pth", "game_jepa_ep6.pth"]:
        candidate = ckpt_dir / name
        if candidate.exists():
            sb_ckpt = str(candidate)
            break

    if not sb_ckpt:
        # Try any epoch checkpoint
        epoch_ckpts = sorted(ckpt_dir.glob("game_jepa_ep*.pth"))
        if epoch_ckpts:
            sb_ckpt = str(epoch_ckpts[-1])

    # Step 4: Train Model A — FROM SCRATCH
    epochs = 20
    lr = 1e-3

    print(f"\n  {'='*50}")
    print(f"  Model A: GraspJEPA FROM SCRATCH")
    print(f"  {'='*50}")
    model_a = GraspJEPA(hidden_dim=512).to(device)
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    print(f"  Trainable params: {params_a:,}")
    t_a = time.time()
    history_a = train_model(model_a, dataloader, epochs, lr, device, "FromScratch")
    time_a = time.time() - t_a

    # Save
    a_path = ckpt_dir / "grasp_jepa_scratch.pth"
    torch.save(model_a.state_dict(), str(a_path))
    print(f"  Saved to {a_path.name} ({time_a:.1f}s)")

    # Step 5: Train Model B — TRANSFER
    if sb_ckpt:
        print(f"\n  {'='*50}")
        print(f"  Model B: GraspJEPA TRANSFER (frozen Predictor)")
        print(f"  Source: {Path(sb_ckpt).name}")
        print(f"  {'='*50}")
        model_b = GraspJEPA.from_transfer(sb_ckpt, device=device)
        params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
        print(f"  Trainable params: {params_b:,} (encoder only)")
        t_b = time.time()
        history_b = train_model(model_b, dataloader, epochs, lr, device, "Transfer")
        time_b = time.time() - t_b

        b_path = ckpt_dir / "grasp_jepa_transfer.pth"
        torch.save(model_b.state_dict(), str(b_path))
        print(f"  Saved to {b_path.name} ({time_b:.1f}s)")
    else:
        print(f"\n  WARNING: No Science Birds checkpoint found for transfer!")
        print(f"  Skipping Model B (Transfer).")
        history_b = None
        time_b = 0

    # Step 6: Compare
    total_time = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  CROSS-DOMAIN TRANSFER RESULTS")
    print(f"  {'='*60}")

    print(f"\n  Model A (From Scratch):")
    print(f"    Final Loss:    {history_a[-1]['loss']:.4f}")
    print(f"    Final CosSim:  {history_a[-1]['cos_sim']:.4f}")
    print(f"    Training time: {time_a:.1f}s")
    print(f"    Trainable:     {params_a:,} params")

    if history_b:
        print(f"\n  Model B (Transfer — frozen Predictor):")
        print(f"    Final Loss:    {history_b[-1]['loss']:.4f}")
        print(f"    Final CosSim:  {history_b[-1]['cos_sim']:.4f}")
        print(f"    Training time: {time_b:.1f}s")
        print(f"    Trainable:     {params_b:,} params")

        # Transfer efficiency
        loss_ratio = history_b[-1]['loss'] / max(history_a[-1]['loss'], 1e-6)
        print(f"\n  Transfer Efficiency:")
        print(f"    Loss ratio (B/A):     {loss_ratio:.3f} (< 1 = transfer better)")
        print(f"    Time ratio (B/A):     {time_b/max(time_a,1):.3f}")
        print(f"    Param ratio (B/A):    {params_b/max(params_a,1):.3f}")

        # Convergence speed: at which epoch did B reach A's final loss?
        a_final = history_a[-1]['loss']
        converge_epoch = None
        for h in history_b:
            if h['loss'] <= a_final:
                converge_epoch = h['epoch']
                break
        if converge_epoch:
            print(f"    Converge to A's final: epoch {converge_epoch}/{epochs}")
        else:
            print(f"    B did not reach A's final loss")

    print(f"\n  Total experiment time: {total_time/60:.1f} min")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "evaluation"
    results_dir.mkdir(exist_ok=True)
    results = {
        "from_scratch": history_a,
        "transfer": history_b,
        "time_scratch": time_a,
        "time_transfer": time_b,
        "params_scratch": params_a,
        "params_transfer": params_b if history_b else 0,
    }
    import json
    results_path = results_dir / "transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
