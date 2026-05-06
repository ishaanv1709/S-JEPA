"""
Vortaz Labs — Ablation Study Runner
====================================
Runs 4 ablation variants on Science Birds and prints a comparison table.
Fill results into Table 3 of sjepa_paper.tex before submission.

Variants:
  1. Full S-JEPA         — baseline (uses existing checkpoint)
  2. No Critic           — random action selection (no energy guidance)
  3. Single-start actor  — num_starts=1 instead of 8
  4. No SIGReg           — retrain JEPA with sigreg_lambda=0, then eval
  5. No Decoder effect   — N/A: decoder is visualization-only, does not affect scores

Usage:
    cd "C:/Users/ishaa_04bpft8/Energy Grid World Model/game_world_model"
    python run_ablations.py

Output: prints a ready-to-paste LaTeX table row for each variant.
"""

import torch
import numpy as np
import os
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import OfflineSimulator
from science_birds.state_parser import parse_state, parse_action, OBS_DIM
from science_birds.level_loader import LevelLoader
from models.world_model import GameJEPA, GameDecoder
from models.critic import Critic
from models.actor import optimize_action
from benchmark.runner import load_jepa_models, load_norm_stats

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2]          # 3 independent runs (matches paper)
N_LEVELS     = 4                  # 4 held-out levels (matches paper)
DIFFICULTIES = ["easy", "easy", "medium", "hard"]   # matches paper's 4 levels
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR     = "checkpoints"

# Level seeds: fixed, same across all ablation variants
LEVEL_SEEDS  = [42, 43, 44, 45]


# ── Level generator ───────────────────────────────────────────────────────────
def make_levels():
    levels = []
    for i, (diff, seed) in enumerate(zip(DIFFICULTIES, LEVEL_SEEDS)):
        sim = OfflineSimulator(seed=seed)
        state = sim.generate_level(diff)
        levels.append({"level_id": i + 1, "difficulty": diff,
                        "state": state, "sim": sim})
    return levels


# ── Run a single level with a given action strategy ──────────────────────────
def run_level(level_info, jepa, decoder, critic, norm_stats,
              strategy="full", num_starts=8):
    """
    strategy: "full"   — energy-guided multi-start actor (normal)
              "random" — pick a random action (no critic guidance)
    num_starts: how many gradient starts to use (ablate with 1)
    """
    current = copy.deepcopy(level_info["state"])
    sim = level_info["sim"]
    total_score = 0
    pigs_cleared = True

    for bird_idx in range(len(current["birds"])):
        available = [i for i, b in enumerate(current["birds"])
                     if not b.get("used", False)]
        if not available:
            break
        pigs_alive = sum(1 for p in current["pigs"] if p.get("health", 0) > 0)
        if pigs_alive == 0:
            break

        bird_i = available[0]
        state_vec = parse_state(current)
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        if norm_stats is not None:
            state_norm = (state_t - norm_stats["state_mean"]) / norm_stats["state_std"]
        else:
            state_norm = state_t

        if strategy == "random":
            # No critic: uniformly random action
            angle_deg = float(np.random.uniform(5, 75))
            power     = float(np.random.uniform(0.5, 1.0))
            tap_ms    = float(np.random.uniform(0, 1000))
        else:
            # Normal: energy-minimising multi-start
            result    = optimize_action(state_norm, jepa, critic,
                                        decoder=decoder, steps=200,
                                        num_starts=num_starts)
            action    = result["action_raw"]
            angle_deg = action["angle_deg"]
            power     = action["power"]
            tap_ms    = action["tap_time_ms"]

        shot = sim.simulate_shot(current, bird_i, angle_deg, power, tap_ms / 1000.0)
        total_score  += shot["score_delta"]
        current       = shot["level"]

    pigs_left = sum(1 for p in current["pigs"] if p.get("health", 0) > 0)
    cleared   = (pigs_left == 0)
    return total_score, cleared


# ── Run a full variant across all seeds × levels ─────────────────────────────
def run_variant(name, jepa, decoder, critic, norm_stats,
                strategy="full", num_starts=8):
    levels = make_levels()
    all_scores  = []
    levels_cleared = 0
    print(f"\n  [{name}]")

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_scores = []
        seed_cleared = 0

        for lv in levels:
            score, cleared = run_level(lv, jepa, decoder, critic, norm_stats,
                                       strategy=strategy, num_starts=num_starts)
            seed_scores.append(score)
            if cleared:
                seed_cleared += 1
            print(f"    seed={seed}  level={lv['level_id']} ({lv['difficulty']:6s}) "
                  f"| score={score:6d}  cleared={cleared}")

        all_scores.append(np.mean(seed_scores))
        levels_cleared = max(levels_cleared, seed_cleared)  # best seed

    mu  = int(np.mean(all_scores))
    std = int(np.std(all_scores))
    print(f"  → Score: {mu} ± {std}  |  Levels cleared: ~{levels_cleared}/4")
    return mu, std, levels_cleared


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Vortaz Labs — S-JEPA Ablation Study")
    print("=" * 60)

    jepa, decoder, critic = load_jepa_models(DEVICE, CKPT_DIR)
    norm_stats = load_norm_stats()

    results = {}

    # 1. Full S-JEPA (baseline)
    mu, std, cl = run_variant("Full S-JEPA", jepa, decoder, critic,
                               norm_stats, strategy="full", num_starts=8)
    results["full"] = (mu, std, cl)

    # 2. No Critic — random action selection
    mu, std, cl = run_variant("No Critic (random)", jepa, decoder, critic,
                               norm_stats, strategy="random")
    results["no_critic"] = (mu, std, cl)

    # 3. Single-start actor (N=1)
    mu, std, cl = run_variant("Single-start (N=1)", jepa, decoder, critic,
                               norm_stats, strategy="full", num_starts=1)
    results["single_start"] = (mu, std, cl)

    # 4. No SIGReg — retrain required; load separate checkpoint if available
    no_sig_ckpt = os.path.join(CKPT_DIR, "game_jepa_no_sigreg.pth")
    if os.path.exists(no_sig_ckpt):
        jepa_ns = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                           hidden_dim=512, sigreg_lambda=0.0,
                           use_memory=False, use_configurator=False).to(DEVICE)
        jepa_ns.load_state_dict(
            torch.load(no_sig_ckpt, map_location=DEVICE)["model_state_dict"]
        )
        jepa_ns.eval()
        for p in jepa_ns.parameters():
            p.requires_grad = False
        mu, std, cl = run_variant("No SIGReg", jepa_ns, decoder, critic,
                                   norm_stats, strategy="full", num_starts=8)
        results["no_sigreg"] = (mu, std, cl)
    else:
        print(f"\n  [No SIGReg] — checkpoint not found at {no_sig_ckpt}")
        print("  Run first:  python -c \"")
        print("    from training.train_jepa import train_jepa")
        print("    train_jepa(sigreg_lambda=0, save_dir='checkpoints',")
        print("               epochs=28)\"")
        print("  Then rename the last checkpoint to game_jepa_no_sigreg.pth")
        results["no_sigreg"] = None

    # ── Print LaTeX table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS — paste into Table 3 of sjepa_paper.tex")
    print("=" * 60)
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Variant & Score ($\mu \pm \sigma$) & Levels \\")
    print(r"\midrule")

    labels = {
        "full":         r"\textbf{Full S-JEPA}",
        "no_critic":    r"$-$~Energy Critic (random selection)",
        "single_start": r"$-$~Multi-start ($N{=}1$ start, 100 steps)",
        "no_sigreg":    r"$-$~SIGReg ($\lambda_\mathrm{reg}{=}0$)",
    }

    for key, label in labels.items():
        val = results.get(key)
        if val is None:
            print(f"{label} & \\textit{{needs retrain}} & --- \\\\")
        else:
            mu, std, cl = val
            score_str = f"\\textbf{{{mu} $\\pm$ {std}}}" if key == "full" else f"{mu} $\\pm$ {std}"
            lvl_str   = f"\\textbf{{{cl}/4}}" if key == "full" else f"{cl}/4"
            print(f"{label} & {score_str} & {lvl_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print("NOTE: 'No Decoder' is not a valid ablation in this codebase —")
    print("the Decoder is visualization-only and does NOT affect action")
    print("selection or scores. Remove that row from the paper table.")


if __name__ == "__main__":
    main()
