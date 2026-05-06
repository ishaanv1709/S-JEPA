"""
Evaluate SY-JEPA, Plain MLP Dynamics, and C-SWM baselines on the offline
Science Birds simulator with state-vector-derived target angles.

Using screenshot angle (actor.py default) would give ALL models the same
constant 31 degree angle — making the comparison trivially identical.
Instead we derive angle from pig positions in the state vector, so each
model's energy critic can still influence power and tap while the angle
comes from the same consistent geometric source.

Runs all 11 benchmark levels (matching benchmark/runner.py) × 3 seeds each.
Reports relative scores (SY-JEPA = 100%) so offline absolute values are
not confused with live Unity scores.

Requires:
  checkpoints/game_jepa_ep28.pth  + checkpoints/game_critic.pth
  checkpoints/baseline_plain_mlp.pth  + checkpoints/baseline_plain_critic.pth
  checkpoints/baseline_cswm.pth       + checkpoints/baseline_cswm_critic.pth

Run:
    python eval_baselines_sim.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import OfflineSimulator
from science_birds.state_parser import parse_state
from models.world_model import GameJEPA
from models.critic import Critic
from training.train_baselines import BaselineModel


N_LEVELS = 11          # matches benchmark/runner.py N_BENCHMARK_LEVELS
N_SEEDS  = 3
N_STARTS = 8
STEPS    = 100

# Matches benchmark/runner.py: difficulties cycle easy/medium/hard, seed = 1000+i
DIFFICULTIES = ["easy", "medium", "hard"]
LEVEL_SEED_OFFSET = 1000   # sim seed = LEVEL_SEED_OFFSET + level_idx (0-based)

# State-vector offsets (see science_birds/state_parser.py)
PIG_BASE     = 140   # Pigs start at index 140
PIG_FEATURES = 4     # size, x, y, health_pct
SLING_X_IDX  = 160
SLING_Y_IDX  = 161
NORM_X = 800.0
NORM_Y = 400.0


# ── Angle from state vector ───────────────────────────────────────────────────

def _angle_from_state(obs_raw: np.ndarray) -> float:
    """Derive normalised launch angle from pig centroid in the state vector."""
    sx = obs_raw[SLING_X_IDX] * NORM_X
    sy = obs_raw[SLING_Y_IDX] * NORM_Y  # screen-Y from top

    pig_xs, pig_ys = [], []
    for i in range(5):
        base   = PIG_BASE + i * PIG_FEATURES
        health = obs_raw[base + 3]
        if health > 0.01:
            pig_xs.append(obs_raw[base + 1] * NORM_X)
            pig_ys.append(obs_raw[base + 2] * NORM_Y)

    if not pig_xs:
        return 31.0 / 90.0   # geometric fallback

    tx = float(np.mean(pig_xs))
    ty = float(np.mean(pig_ys))

    dx = tx - sx               # rightward
    dy = sy - ty               # upward (screen-Y is inverted)

    los  = np.degrees(np.arctan2(dy, dx))
    dist = np.sqrt(dx**2 + dy**2)
    grav = float(np.clip(dist / 50.0, 5.0, 20.0))

    angle = float(np.clip(los + grav, 10.0, 75.0))
    return angle / 90.0


# ── Norm stats ────────────────────────────────────────────────────────────────

def load_norm_stats(device: str):
    """Load GameDataset Z-score stats used during training."""
    path = Path(__file__).resolve().parent / "data" / "norm_stats.npz"
    if not path.exists():
        return None
    s = np.load(str(path))
    return {
        "state_mean": torch.tensor(s["state_mean"], dtype=torch.float32).to(device),
        "state_std":  torch.tensor(s["state_std"],  dtype=torch.float32).to(device),
        "action_mean": torch.tensor(s["action_mean"], dtype=torch.float32).to(device),
        "action_std":  torch.tensor(s["action_std"],  dtype=torch.float32).to(device),
    }


# ── Action optimiser ──────────────────────────────────────────────────────────

def _optimize_action(encoder, predictor, critic, obs_raw: np.ndarray,
                     device: str, norm_stats: dict,
                     n_starts: int = N_STARTS, steps: int = STEPS) -> tuple:
    """
    Multi-start gradient actor.
    Angle: locked to state-vector target (±5 deg band).
    Power + tap: optimised by critic energy.

    obs_raw is the raw [0,1] parse_state() vector.
    Z-score normalization is applied before the encoder/predictor/critic
    so all models receive inputs matching their training distribution.
    Returns (angle_deg, power, tap_sec).
    """
    # Z-score normalize state for encoder (angle computation uses raw obs_raw)
    state_t = torch.tensor(obs_raw, dtype=torch.float32).unsqueeze(0).to(device)
    if norm_stats is not None:
        state_norm = (state_t - norm_stats["state_mean"]) / norm_stats["state_std"]
    else:
        state_norm = state_t

    with torch.no_grad():
        z_t = encoder(state_norm)

    target_a  = float(np.clip(_angle_from_state(obs_raw), 0.05, 0.95))
    angle_lo  = max(0.0, target_a - 0.055)
    angle_hi  = min(1.0, target_a + 0.055)

    starts = [
        (target_a,        1.00, 0.00),
        (target_a,        0.90, 0.00),
        (target_a,        0.80, 0.00),
        (target_a,        0.70, 0.00),
        (target_a,        0.95, 0.30),
        (target_a,        0.85, 0.50),
        (target_a - 0.04, 0.95, 0.00),
        (target_a + 0.04, 0.90, 0.00),
    ]

    act_lo   = torch.tensor([[angle_lo, 0.0, 0.0]], device=device)
    act_hi   = torch.tensor([[angle_hi, 1.0, 1.0]], device=device)
    hard_lo  = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    hard_hi  = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    # Z-score stats for action (kept as tensors for differentiable normalization)
    if norm_stats is not None:
        a_mean = norm_stats["action_mean"]   # [3]
        a_std  = norm_stats["action_std"]    # [3]
    else:
        a_mean = torch.zeros(3, device=device)
        a_std  = torch.ones(3, device=device)

    best_energy = float("inf")
    best_action = np.array([target_a, 0.85, 0.0])

    for idx in range(min(n_starts, len(starts))):
        a0, p0, t0 = starts[idx]
        a0 = float(np.clip(a0, 0.05, 0.95))

        # act stays in [0,1] space for clamping; Z-score applied when calling model
        act = nn.Parameter(
            torch.tensor([[a0, p0, t0]], dtype=torch.float32, device=device)
        )
        opt = optim.Adam([act], lr=0.01)

        for _ in range(steps):
            opt.zero_grad()
            with torch.no_grad():
                act.data = torch.clamp(act.data, act_lo, act_hi)
            # Z-score the action before passing to predictor/critic (differentiable)
            act_z  = (act - a_mean) / a_std
            z_pred = predictor(z_t, act_z)
            e      = critic(z_t, z_pred, act_z).mean()
            e.backward()
            opt.step()
            with torch.no_grad():
                act.data = torch.clamp(act.data, act_lo, act_hi)

        with torch.no_grad():
            act.data = torch.clamp(act.data, hard_lo, hard_hi)
            act_z    = (act - a_mean) / a_std
            z_pred   = predictor(z_t, act_z)
            final_e  = critic(z_t, z_pred, act_z).item()

        if final_e < best_energy:
            best_energy = final_e
            best_action = act.detach().cpu().numpy()[0]

    angle = float(best_action[0] * 90.0)
    power = float(np.clip(best_action[1], 0.0, 1.0))
    tap   = float(best_action[2] * 3.0)
    return angle, power, tap


# ── Game loop ─────────────────────────────────────────────────────────────────

def play_one_game(sim, level_state: dict, encoder, predictor, critic,
                  device: str, norm_stats: dict) -> int:
    level = copy.deepcopy(level_state)
    total = 0

    for _ in range(len(level["birds"])):
        available = [i for i, b in enumerate(level["birds"])
                     if not b.get("used", False)]
        if not available:
            break
        if sum(1 for p in level["pigs"] if p.get("health", 0) > 0) == 0:
            break

        obs_raw         = parse_state(level)
        angle, pwr, tap = _optimize_action(encoder, predictor, critic,
                                           obs_raw, device, norm_stats)
        result          = sim.simulate_shot(level, available[0], angle, pwr, tap)
        total          += result["score_delta"]
        level           = result["level"]

    return total


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(name: str, encoder, predictor, critic,
              device: str, norm_stats: dict) -> dict:
    """Run model on all 11 benchmark levels × N_SEEDS seeds."""
    level_scores = {}
    for i in range(N_LEVELS):
        lvl  = i + 1
        diff = DIFFICULTIES[i % 3]
        seed_scores = []
        for seed_offset in range(N_SEEDS):
            sim   = OfflineSimulator(seed=LEVEL_SEED_OFFSET + i + seed_offset * 100)
            state = sim.generate_level(diff)
            score = play_one_game(sim, state, encoder, predictor, critic,
                                  device, norm_stats)
            seed_scores.append(score)
        mu = float(np.mean(seed_scores))
        level_scores[lvl] = mu
        print(f"  Level {lvl:>2} ({diff:6s}) mean: {mu:.1f}")

    total = sum(level_scores.values())
    print(f"  {name} total across 11 levels: {total:.1f}\n")
    return {"level_scores": {str(k): v for k, v in level_scores.items()},
            "total": total}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_jepa(device: str):
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"

    jepa = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                    hidden_dim=512, use_memory=False,
                    use_configurator=False).to(device)
    path = ckpt_dir / "game_jepa_ep28.pth"
    if not path.exists():
        print(f"  SKIP SY-JEPA: {path} not found")
        return None, None, None

    state = torch.load(str(path), map_location=device)
    jepa.load_state_dict(state["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    cpath  = ckpt_dir / "game_critic.pth"
    if cpath.exists():
        critic.load_state_dict(torch.load(str(cpath), map_location=device))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    # Thin wrappers so the API matches BaselineModel
    class _Enc(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m.encoder(x)

    class _Pred(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, z, a):
            return self.m.predictor(z, a)

    return _Enc(jepa).to(device), _Pred(jepa).to(device), critic


def load_baseline(enc_name: str, critic_name: str, device: str):
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"

    ep = ckpt_dir / enc_name
    if not ep.exists():
        print(f"  SKIP: {ep} not found. Run training/train_baselines.py first.")
        return None, None, None

    bm = BaselineModel().to(device)
    bm.load_state_dict(torch.load(str(ep), map_location=device)["model_state_dict"])
    bm.eval()
    for p in bm.parameters():
        p.requires_grad = False

    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    cp = ckpt_dir / critic_name
    if cp.exists():
        critic.load_state_dict(torch.load(str(cp), map_location=device))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    class _Enc(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m.encode(x)

    class _Pred(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, z, a):
            return self.m.predict(z, a)

    return _Enc(bm).to(device), _Pred(bm).to(device), critic


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Offline Sim — SY-JEPA vs World Model Baselines")
    print("  (State-vector angle targeting, N=8 multi-start actor)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    norm_stats = load_norm_stats(device)
    if norm_stats is None:
        print("  WARNING: data/norm_stats.npz not found — running without Z-score norm")
    else:
        print("  Loaded Z-score norm stats from data/norm_stats.npz\n")

    model_defs = [
        ("SY-JEPA",       lambda: load_jepa(device)),
        ("Plain MLP",     lambda: load_baseline("baseline_plain_mlp.pth",
                                                "baseline_plain_critic.pth",
                                                device)),
        ("C-SWM",         lambda: load_baseline("baseline_cswm.pth",
                                                "baseline_cswm_critic.pth",
                                                device)),
    ]

    results   = {}
    ref_total = None

    for name, loader in model_defs:
        enc, pred, crit = loader()
        if enc is None:
            continue
        print(f"  Running {name}...")
        results[name] = run_model(name, enc, pred, crit, device, norm_stats)
        if name == "SY-JEPA":
            ref_total = results[name]["total"]

    # ── Summary table ─────────────────────────────────────────────────────────
    ref = ref_total if ref_total else 1.0
    print("=" * 60)
    print("  RESULTS (relative to SY-JEPA = 100%)")
    print("=" * 60)
    print(f"  {'Model':<22} {'Total':>10} {'Rel. Score':>12}")
    print(f"  {'-'*46}")
    for name, r in results.items():
        rel    = r["total"] / ref * 100.0 if ref else 0.0
        marker = " ←" if name == "SY-JEPA" else ""
        print(f"  {name:<22} {r['total']:>10.1f} {rel:>11.1f}%{marker}")
    print()

    out = Path(__file__).resolve().parent / "sim_baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out}")


if __name__ == "__main__":
    main()
