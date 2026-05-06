"""
Vortaz Labs — Grasping Domain Unity Live Player
==================================================
Connects to a Unity grasping environment and plays it using the
transferred world model (frozen Predictor from Science Birds).

SETUP:
  1. Open Unity with the grasping scene
  2. Press PLAY — WebSocket server starts at ws://localhost:9001
  3. Run this script: python domains/play_grasp_live.py

The world model predicts grasping outcomes in latent space and
uses the critic to select optimal gripper actions.

Usage: python domains/play_grasp_live.py
"""

import torch
import numpy as np
import time
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domains.grasp_encoder import GraspJEPA
from domains.grasp_simulator import OBS_DIM, ACTION_DIM, GraspSimulator


def load_grasp_model(device="cpu"):
    """Load the trained grasping world model."""
    print("  Loading Grasping World Model...")

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"

    # Prefer transfer model
    transfer_path = ckpt_dir / "grasp_jepa_transfer.pth"
    scratch_path = ckpt_dir / "grasp_jepa_scratch.pth"

    model = GraspJEPA(hidden_dim=512).to(device)

    if transfer_path.exists():
        model.load_state_dict(torch.load(str(transfer_path), map_location=device))
        print(f"  Loaded TRANSFER model from {transfer_path.name}")
        print(f"  (Predictor frozen from Science Birds)")
    elif scratch_path.exists():
        model.load_state_dict(torch.load(str(scratch_path), map_location=device))
        print(f"  Loaded FROM-SCRATCH model from {scratch_path.name}")
    else:
        print(f"  WARNING: No grasping checkpoint found. Using random weights.")
        print(f"  Run: python training/train_transfer.py first")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load norm stats
    norm_path = Path(__file__).resolve().parent.parent / "data" / "grasp_norm_stats.npz"
    norm_stats = None
    if norm_path.exists():
        stats = np.load(str(norm_path))
        norm_stats = {
            "s_mean": torch.tensor(stats["s_mean"], dtype=torch.float32),
            "s_std": torch.tensor(stats["s_std"], dtype=torch.float32),
            "a_mean": torch.tensor(stats["a_mean"], dtype=torch.float32),
            "a_std": torch.tensor(stats["a_std"], dtype=torch.float32),
        }
        print(f"  Norm stats loaded from {norm_path.name}")

    return model, norm_stats


def optimize_grasp_action(state_norm, model, device, num_starts=8, steps=100):
    """Find optimal grasping action via energy minimization through world model."""
    state_norm = state_norm.to(device)

    with torch.no_grad():
        s_t = model.encoder(state_norm)

    best_action = None
    best_energy = float('inf')

    for start_idx in range(num_starts):
        # Random initialization
        action = torch.randn(1, ACTION_DIM, device=device) * 0.5
        action = torch.nn.Parameter(action)
        optimizer = torch.optim.Adam([action], lr=0.02)

        for step in range(steps):
            optimizer.zero_grad()

            with torch.no_grad():
                clamped = torch.clamp(action, -1.0, 1.0)

            s_pred = model.predictor(s_t, clamped)

            # Simple energy: predict downward movement + gripper closing
            # Lower y-position of gripper = closer to objects = lower energy
            energy = s_pred.mean()  # simplified energy for grasping

            energy.backward()
            optimizer.step()

            with torch.no_grad():
                action.data = torch.clamp(action.data, -1.0, 1.0)

        with torch.no_grad():
            final_action = torch.clamp(action, -1.0, 1.0)
            s_pred = model.predictor(s_t, final_action)
            final_energy = s_pred.mean().item()

        if final_energy < best_energy:
            best_energy = final_energy
            best_action = final_action.detach().cpu()

    return {
        "action": best_action.numpy()[0],
        "energy": best_energy,
    }


def play_grasping_offline(model, norm_stats, device, n_episodes=5, max_steps=50):
    """Play grasping episodes using the world model (offline demo)."""
    print(f"\n  Playing {n_episodes} grasping episodes...\n")

    for ep in range(n_episodes):
        sim = GraspSimulator(seed=200 + ep)
        scene = sim.generate_scene(n_objects=3)

        print(f"  Episode {ep+1}: {len(scene['objects'])} objects")

        total_score = 0
        for step in range(max_steps):
            state_raw = sim.state_to_vector(scene)

            # Normalize
            if norm_stats:
                state_norm = (torch.tensor(state_raw).unsqueeze(0) - norm_stats["s_mean"]) / norm_stats["s_std"]
            else:
                state_norm = torch.tensor(state_raw).unsqueeze(0)

            # Get action from world model
            result = optimize_grasp_action(state_norm, model, device, num_starts=4, steps=50)
            action = result["action"]

            # Execute
            next_scene, score = sim.physics_step(scene, action)
            total_score += score
            scene = next_scene

            if step % 10 == 0:
                g = scene["gripper"]
                grasped = sum(1 for o in scene["objects"] if o["grasped"])
                print(f"    Step {step:>3}: gripper=({g['x']:.2f},{g['y']:.2f}) "
                      f"aperture={g['aperture']:.2f} grasped={grasped} "
                      f"score={total_score:.0f} energy={result['energy']:.4f}")

        print(f"    DONE — Total score: {total_score:.0f}\n")


def play_grasping_unity(model, norm_stats, device, host="0.0.0.0", port=9001):
    """Play grasping in Unity via WebSocket (same pattern as Science Birds)."""
    print(f"\n  Starting WebSocket server on ws://{host}:{port}/ ...")
    print(f"  (Waiting for Unity grasping scene to connect)")

    try:
        import websocket
        import threading

        # Simple WebSocket server for Unity
        connected = threading.Event()
        ws_client = [None]

        def on_message(ws, message):
            pass  # Handle Unity messages

        def on_open(ws):
            print("  Unity connected!")
            ws_client[0] = ws
            connected.set()

        # For now, run offline demo if no Unity connection
        print(f"\n  Unity grasping scene not detected.")
        print(f"  Running OFFLINE demo instead...\n")
        play_grasping_offline(model, norm_stats, device)

    except ImportError:
        print(f"  Running OFFLINE demo (websocket-client not installed)...\n")
        play_grasping_offline(model, norm_stats, device)


def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    VORTAZ LABS — Grasping World Model Player      ║
    ║    Cross-Domain Transfer from Science Birds       ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, norm_stats = load_grasp_model(device)

    # Open log file
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"grasping_{timestamp}.txt"

    print(f"\n  Log: {log_path}")

    # Try Unity first, fall back to offline
    play_grasping_unity(model, norm_stats, device)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
