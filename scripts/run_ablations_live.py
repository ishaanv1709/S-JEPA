"""
Vortaz Labs — Live Unity Ablation Study (per-level mode)
=========================================================
Load a level in Unity first, then run this script for that level.
Results accumulate in ablation_results.json.

Usage — standard 3 variants (full, no_critic, single_start):
  python run_ablations_live.py --level 1

Usage — SIGReg ablation (run AFTER retrain_no_sigreg.py):
  python run_ablations_live.py --level 1 --no-sigreg

Usage — world model baselines (run AFTER train_baselines.py + train_baseline_critics.py):
  python run_ablations_live.py --level 1 --baselines
  python run_ablations_live.py --level 2 --baselines
  python run_ablations_live.py --level 3 --baselines
  python run_ablations_live.py --level 4 --baselines

Final table (after all passes):
  python run_ablations_live.py --report
"""

import sys
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import ScienceBirdsClient
from models.actor import _compute_target_angle_from_screenshot
from models.actor import optimize_action
from play_live import load_models, build_state_from_screenshot

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS    = 3      # runs per variant per level
MAX_SHOTS = 5
SHOT_WAIT = 3.0
RESULTS_FILE = Path("ablation_results.json")

VARIANTS = [
    ("full",         "Full S-JEPA",              {"num_starts": 8}),
    ("no_critic",    "No Critic (random action)", {"num_starts": 0}),
    ("single_start", "Single-start (N=1)",        {"num_starts": 1}),
]

# SIGReg ablation: same as full but with no-sigreg checkpoint
SIGREG_VARIANT = ("no_sigreg", "No SIGReg (retrained)", {"num_starts": 8})

# World model baselines (Plain MLP + C-SWM), each with their own critic
BASELINE_VARIANTS = [
    ("plain_mlp", "Plain MLP Dynamics",
     {"num_starts": 8,
      "encoder_ckpt": "checkpoints/baseline_plain_mlp.pth",
      "critic_ckpt":  "checkpoints/baseline_plain_critic.pth"}),
    ("cswm",      "C-SWM (Kipf et al.)",
     {"num_starts": 8,
      "encoder_ckpt": "checkpoints/baseline_cswm.pth",
      "critic_ckpt":  "checkpoints/baseline_cswm_critic.pth"}),
]


# ── Baseline model loader ─────────────────────────────────────────────────────
def load_baseline(encoder_ckpt, critic_ckpt, device):
    """Load a baseline encoder+predictor and its critic."""
    from training.train_baselines import BaselineModel
    from models.critic import Critic

    baseline = BaselineModel().to(device)
    baseline.load_state_dict(
        torch.load(encoder_ckpt, map_location=device)["model_state_dict"]
    )
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    critic.load_state_dict(torch.load(critic_ckpt, map_location=device))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    return baseline, critic


def baseline_optimize_action(state_norm, baseline, critic,
                              norm_stats, state_raw, num_starts=8):
    """
    Same multi-start gradient actor as SY-JEPA but running through
    a baseline (Plain MLP or C-SWM) encoder+predictor.
    """
    import torch.optim as optim
    from models.actor import _compute_target_angle_from_screenshot

    device = next(baseline.parameters()).device

    # Vision-based angle (same as SY-JEPA)
    angle_norm = _compute_target_angle_from_screenshot("screenshot_debug.png")
    angle_init = float(np.clip(angle_norm * 90.0, 5.0, 75.0))

    z_t = baseline.encode(state_norm.to(device))

    best_energy = float("inf")
    best_action = None

    for _ in range(num_starts):
        # Random start around vision angle ± 10°
        angle_start = np.clip(angle_init + np.random.uniform(-10, 10), 5, 75)
        a_raw = torch.tensor([
            [angle_start / 90.0,
             np.random.uniform(0.6, 1.0),
             np.random.uniform(0.0, 1.0)]
        ], dtype=torch.float32, device=device, requires_grad=True)

        opt = optim.Adam([a_raw], lr=0.01)
        for _ in range(100):
            opt.zero_grad()
            a_clamped = torch.clamp(a_raw, 0.0, 1.0)
            z_pred    = baseline.predict(z_t, a_clamped)
            energy    = critic(z_pred).mean()
            energy.backward()
            opt.step()

        with torch.no_grad():
            a_final  = torch.clamp(a_raw, 0.0, 1.0)
            z_pred   = baseline.predict(z_t, a_final)
            e_final  = critic(z_pred).item()
            if e_final < best_energy:
                best_energy = e_final
                best_action = a_final.squeeze().cpu().numpy()

    angle_deg = float(best_action[0] * 90.0)
    power     = float(best_action[1])
    tap_ms    = float(best_action[2] * 1000.0)
    return angle_deg, power, tap_ms, best_energy


# ── Action chooser ─────────────────────────────────────────────────────────────
def choose_action(screenshot, state_raw, norm_stats,
                  jepa, critic, decoder, device, num_starts):
    if num_starts == 0:
        angle_norm = _compute_target_angle_from_screenshot("screenshot_debug.png")
        angle = float(np.clip(angle_norm * 90.0, 5.0, 75.0))
        angle += float(np.random.uniform(-5.0, 5.0))
        angle = float(np.clip(angle, 5.0, 75.0))
        power  = float(np.random.uniform(0.6, 1.0))
        tap_ms = float(np.random.uniform(0.0, 800.0))
        return angle, power, tap_ms, float("nan")

    if norm_stats:
        state_norm = ((torch.tensor(state_raw).unsqueeze(0) -
                       norm_stats["state_mean"]) /
                      norm_stats["state_std"])
    else:
        state_norm = torch.tensor(state_raw).unsqueeze(0)

    result = optimize_action(
        state_norm, jepa, critic,
        decoder=decoder, steps=100, num_starts=num_starts,
        obs_raw=state_raw
    )
    a = result["action_raw"]
    return a["angle_deg"], a["power"], a["tap_time_ms"], result["energy"]


# ── Play one run of current level ─────────────────────────────────────────────
def play_one_run(client, jepa, decoder, critic,
                 norm_stats, device, num_starts, run_idx,
                 baseline_models=None):
    """Play whatever level is currently loaded in Unity. Returns (score, cleared)."""
    total_score = 0
    cleared     = False

    for shot_num in range(MAX_SHOTS):
        state = client.get_state()
        if state == client.STATE_WON:
            cleared = True
            break
        if state not in (client.STATE_PLAYING, "GameWorld"):
            break

        screenshot = client.get_screenshot()
        if screenshot:
            with open("screenshot_debug.png", "wb") as f:
                f.write(screenshot)

        pre_score = client.get_score()
        state_raw = build_state_from_screenshot(screenshot, pre_score, shot_num)

        if baseline_models is not None:
            bl_enc, bl_crit = baseline_models
            if norm_stats:
                sn = ((torch.tensor(state_raw).unsqueeze(0) -
                       norm_stats["state_mean"]) / norm_stats["state_std"])
            else:
                sn = torch.tensor(state_raw).unsqueeze(0)
            angle, power, tap_ms, energy = baseline_optimize_action(
                sn, bl_enc, bl_crit, norm_stats, state_raw,
                num_starts=num_starts
            )
        else:
            angle, power, tap_ms, energy = choose_action(
                screenshot, state_raw, norm_stats,
                jepa, critic, decoder, device, num_starts
            )

        e_str = f"{energy:.4f}" if not np.isnan(float(energy)) else "random"
        print(f"      Shot {shot_num+1}: {angle:.1f}° {power:.0%} tap={tap_ms:.0f}ms e={e_str}")

        client.do_shot_polar(angle, power, tap_ms / 1000.0)
        time.sleep(SHOT_WAIT)

        total_score = client.get_score()
        delta = total_score - pre_score
        print(f"      → +{delta} pts (total {total_score})")

    if client.get_state() == client.STATE_WON:
        cleared = True

    return total_score, cleared


# ── Run one level, all variants ───────────────────────────────────────────────
def run_level(level_id, client, jepa, decoder, critic, norm_stats, device,
              variants_override=None, baseline_model_map=None):
    level_results = {}
    variants = variants_override if variants_override is not None else VARIANTS

    for v_key, v_name, v_cfg in variants:
        print(f"\n  [{v_name}]")
        scores, clears = [], []

        for run_idx in range(N_RUNS):
            np.random.seed(run_idx)
            torch.manual_seed(run_idx)

            print(f"    Run {run_idx+1}/{N_RUNS} — reload level {level_id} in Unity, press Enter")
            input("    (press Enter when level is loaded and ready) > ")

            bl_models = (baseline_model_map or {}).get(v_key, None)
            score, cleared = play_one_run(
                client, jepa, decoder, critic,
                norm_stats, device,
                v_cfg["num_starts"], run_idx,
                baseline_models=bl_models
            )
            scores.append(score)
            clears.append(int(cleared))
            print(f"    Run {run_idx+1} done: score={score} cleared={cleared}")

        level_results[v_key] = {
            "scores":  scores,
            "cleared": clears,
            "mu":      int(np.mean(scores)),
            "std":     int(np.std(scores)),
            "cleared_pct": float(np.mean(clears)),
        }
        print(f"  [{v_name}] level {level_id}: {level_results[v_key]['mu']} ± {level_results[v_key]['std']}")

    return level_results


# ── Report ────────────────────────────────────────────────────────────────────
def print_report():
    if not RESULTS_FILE.exists():
        print("No ablation_results.json found. Run levels first.")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    levels_done = sorted(data.keys(), key=int)
    print(f"\nLevels collected: {[int(l) for l in levels_done]}")
    if len(levels_done) < 4:
        print(f"WARNING: only {len(levels_done)}/4 levels done — run missing levels first.")

    # Aggregate across levels
    agg = {}
    all_variant_keys = ([k for k,_,_ in VARIANTS] + ["no_sigreg"]
                        + [k for k,_,_ in BASELINE_VARIANTS])
    for v_key in all_variant_keys:
        all_scores = []
        all_clears = []
        for lv in levels_done:
            r = data[lv].get(v_key, {})
            all_scores.extend(r.get("scores", []))
            all_clears.extend(r.get("cleared", []))
        if all_scores:
            agg[v_key] = {
                "mu":      int(np.mean(all_scores)),
                "std":     int(np.std(all_scores)),
                "cleared": sum(all_clears),
                "total":   len(all_clears),
            }

    print("\n" + "="*60)
    print("  ABLATION RESULTS — paste into Table 3 of sjepa_paper.tex")
    print("="*60 + "\n")

    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Variant & Score ($\mu \pm \sigma$) & Levels \\")
    print(r"\midrule")

    labels = {
        "full":         r"\textbf{Full S-JEPA}",
        "no_critic":    r"$-$~Energy Critic (random action)",
        "single_start": r"$-$~Single-start actor ($N{=}1$)",
    }

    for v_key in ["full", "no_critic", "single_start"]:
        if v_key not in agg:
            continue
        r = agg[v_key]
        cl_str = f"{r['cleared']}/{r['total']}"
        if v_key == "full":
            print(f"  {labels[v_key]} & \\textbf{{{r['mu']} $\\pm$ {r['std']}}} & \\textbf{{{cl_str}}} \\\\")
        else:
            print(f"  {labels[v_key]} & {r['mu']} $\\pm$ {r['std']} & {cl_str} \\\\")

    if "no_sigreg" in agg:
        r = agg["no_sigreg"]
        cl_str = f"{r['cleared']}/{r['total']}"
        print(f"  $-$~SIGReg ($\\lambda_\\mathrm{{reg}}{{=}}0$) & {r['mu']} $\\pm$ {r['std']} & {cl_str} \\\\")
    else:
        print(r"  $-$~SIGReg ($\lambda_\mathrm{reg}{=}0$) & in prep. & --- \\")

    print(r"\midrule")
    for v_key, v_name, _ in BASELINE_VARIANTS:
        if v_key in agg:
            r = agg[v_key]
            cl_str = f"{r['cleared']}/{r['total']}"
            print(f"  {v_name} & {r['mu']} $\\pm$ {r['std']} & {cl_str} \\\\")
        else:
            print(f"  {v_name} & in prep. & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",     type=int, choices=[1,2,3,4],
                        help="Level ID to run (load this level in Unity first)")
    parser.add_argument("--no-sigreg",  action="store_true",
                        help="Run SIGReg ablation (requires retrain_no_sigreg.py first)")
    parser.add_argument("--baselines",  action="store_true",
                        help="Run Plain MLP + C-SWM baselines (requires train_baselines.py first)")
    parser.add_argument("--report",    action="store_true",
                        help="Print final LaTeX table from saved results")
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if not args.level:
        parser.print_help()
        return

    level_id   = args.level
    no_sigreg  = getattr(args, "no_sigreg",  False)
    run_baselines = getattr(args, "baselines", False)

    if run_baselines:
        missing = [v["encoder_ckpt"] for _,_,v in BASELINE_VARIANTS
                   if not Path(v["encoder_ckpt"]).exists()]
        if missing:
            print("ERROR: baseline checkpoints not found:", missing)
            print("Run first:")
            print("  python -m training.train_baselines")
            print("  python -m training.train_baseline_critics")
            return
        variants_to_run = BASELINE_VARIANTS
        print(f"\nVortaz Labs Ablation — Level {level_id} — World Model Baselines")
    elif no_sigreg:
        no_sigreg_ckpt = Path("checkpoints/game_jepa_no_sigreg.pth")
        if not no_sigreg_ckpt.exists():
            print("ERROR: checkpoints/game_jepa_no_sigreg.pth not found.")
            print("Run first:  python retrain_no_sigreg.py  (~26 min)")
            return
        variants_to_run = [SIGREG_VARIANT]
        print(f"\nVortaz Labs Ablation — Level {level_id} — SIGReg variant")
    else:
        variants_to_run = VARIANTS
        print(f"\nVortaz Labs Ablation — Level {level_id}")
        print(f"Running {len(VARIANTS)} variants × {N_RUNS} runs")

    print()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # For baseline runs, load baseline models instead of SY-JEPA
    baseline_model_map = {}   # v_key -> (BaselineModel, Critic)
    if run_baselines:
        jepa, decoder, critic, norm_stats = load_models(device)  # for norm_stats only
        for v_key, v_name, v_cfg in BASELINE_VARIANTS:
            bl, bl_crit = load_baseline(
                v_cfg["encoder_ckpt"], v_cfg["critic_ckpt"], device
            )
            baseline_model_map[v_key] = (bl, bl_crit)
            print(f"  Loaded {v_name}")

    # For SIGReg ablation, override the JEPA checkpoint
    if no_sigreg:
        # Temporarily swap the checkpoint by patching load_models
        import play_live as _pl
        _orig_best = None
        # Monkey-patch: make load_models pick up no_sigreg checkpoint
        _orig_glob = Path.glob
        jepa, decoder, critic, norm_stats = load_models(device)
        # Now reload JEPA with no-sigreg weights
        import torch as _torch
        from models.world_model import GameJEPA as _GameJEPA
        _ns_jepa = _GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                             hidden_dim=512, use_memory=False,
                             use_configurator=False).to(device)
        _ns_jepa.load_state_dict(
            _torch.load(str(no_sigreg_ckpt), map_location=device)["model_state_dict"]
        )
        _ns_jepa.eval()
        for p in _ns_jepa.parameters():
            p.requires_grad = False
        jepa = _ns_jepa
        print(f"  JEPA (no SIGReg): {no_sigreg_ckpt}")
    else:
        jepa, decoder, critic, norm_stats = load_models(device)

    print("\nConnecting to Science Birds WebSocket...")
    client = ScienceBirdsClient(host="0.0.0.0", port=9000)
    if not client.connect(timeout=30):
        print("ERROR: Unity did not connect in 30s. Make sure it's in Play mode.")
        return

    print(f"Connected! Starting ablation for Level {level_id}.\n")
    print(f"NOTE: The script will pause before each run and ask you to reload")
    print(f"      the level in Unity. Just navigate to Level {level_id} and press Enter.\n")

    level_results = run_level(
        level_id, client, jepa, decoder, critic, norm_stats, device,
        variants_override=variants_to_run,
        baseline_model_map=baseline_model_map,
    )

    client.disconnect()

    # Save / merge results
    data = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    data[str(level_id)] = level_results
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Show partial report
    print(f"\n--- Level {level_id} summary ---")
    for v_key, v_name, _ in variants_to_run:
        r = level_results[v_key]
        print(f"  {v_name}: {r['mu']} ± {r['std']}")

    levels_done = list(data.keys())
    remaining = [l for l in ["1","2","3","4"] if l not in levels_done]
    next_flag  = "--no-sigreg" if no_sigreg else ""
    if remaining:
        print(f"\nLevels remaining: {remaining}")
        print(f"Next: load Level {remaining[0]} in Unity, then:")
        print(f"  python run_ablations_live.py --level {remaining[0]} {next_flag}".strip())
    else:
        print("\nAll 4 levels done! Generate the final table:")
        print("  python run_ablations_live.py --report")


if __name__ == "__main__":
    main()
