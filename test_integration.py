"""
Quick integration test: screenshot -> state vector -> JEPA encode -> critic optimize -> action
Tests the full pipeline WITHOUT needing Unity running.
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from play_live import build_state_from_screenshot, load_models
from models.actor import optimize_action


def main():
    print("=" * 60)
    print("  INTEGRATION TEST: Screenshot -> World Model -> Action")
    print("=" * 60)

    # 1. Load screenshot
    screenshot_path = Path("screenshot_debug.png")
    if not screenshot_path.exists():
        print("ERROR: screenshot_debug.png not found")
        return
    screenshot_bytes = screenshot_path.read_bytes()
    print(f"\n[1] Screenshot loaded: {len(screenshot_bytes)} bytes")

    # 2. Build state from screenshot
    print("\n[2] Building state vector from screenshot...")
    state_raw = build_state_from_screenshot(screenshot_bytes, score=0, shot_num=0)
    print(f"  State shape: {state_raw.shape}")
    print(f"  State range: [{state_raw.min():.4f}, {state_raw.max():.4f}]")
    print(f"  Non-zero elements: {np.count_nonzero(state_raw)} / {len(state_raw)}")

    # Show key state elements
    print(f"\n  Slingshot pos (norm): ({state_raw[160]:.3f}, {state_raw[161]:.3f})")
    print(f"  Birds remaining (norm): {state_raw[163]:.3f}")

    # Show detected pigs
    pig_offset = 140
    for i in range(5):
        off = pig_offset + i * 4
        if state_raw[off + 3] > 0:  # health > 0 means pig exists
            print(f"  Pig {i+1}: x={state_raw[off+1]*800:.0f}, y={state_raw[off+2]*400:.0f}")

    # Show detected blocks
    block_offset = 20
    n_blocks = 0
    for i in range(20):
        off = block_offset + i * 6
        if state_raw[off + 5] > 0:
            n_blocks += 1
    print(f"  Blocks detected: {n_blocks}")

    # 3. Load v2 models
    print("\n[3] Loading v2 models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa, decoder, critic, norm_stats = load_models(device)

    # 4. Normalize state
    print("\n[4] Normalizing state...")
    state_tensor = torch.tensor(state_raw).unsqueeze(0)
    if norm_stats:
        state_norm = (state_tensor - norm_stats['state_mean']) / norm_stats['state_std']
    else:
        state_norm = state_tensor
    print(f"  Normalized range: [{state_norm.min():.4f}, {state_norm.max():.4f}]")

    # 5. Run actor optimization (the big test)
    print("\n[5] Running actor optimization (8 starts x 100 steps)...")
    result = optimize_action(
        state_norm, jepa, critic,
        decoder=decoder, steps=100, num_starts=8,
        obs_raw=state_raw
    )

    action = result["action_raw"]
    print(f"\n  === OPTIMAL ACTION (critic-decided) ===")
    print(f"  Angle:    {action['angle_deg']:.1f} deg")
    print(f"  Power:    {action['power']:.1%}")
    print(f"  Tap time: {action['tap_time_ms']:.0f} ms")
    print(f"  Energy:   {result['energy']:.6f}")

    # 6. Verify the critic produces different energies for different angles
    print("\n[6] Critic energy landscape test (angle sweep)...")
    with torch.no_grad():
        s_t, _ = jepa.encode(state_norm.to(device))

    test_angles = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    energies = []
    for angle_norm in test_angles:
        test_action = torch.tensor([[angle_norm, 0.85, 0.0]], device=device)
        with torch.no_grad():
            s_pred = jepa.predict(s_t, test_action)
            energy = critic(s_t, s_pred, test_action).item()
        energies.append(energy)
        angle_deg = angle_norm * 90
        print(f"  {angle_deg:5.1f} deg -> energy {energy:.6f}")

    energy_range = max(energies) - min(energies)
    best_idx = np.argmin(energies)
    best_angle = test_angles[best_idx] * 90

    print(f"\n  Energy range: {energy_range:.6f}")
    print(f"  Best angle from sweep: {best_angle:.1f} deg")
    print(f"  Optimizer chose:       {action['angle_deg']:.1f} deg")

    if energy_range > 0.01:
        print("\n  PASS: Critic has meaningful energy landscape (range > 0.01)")
    else:
        print("\n  WARN: Critic energy range is small — may need more training")

    if abs(action['angle_deg']) > 1.0:
        print("  PASS: Optimizer chose non-zero angle")
    else:
        print("  WARN: Optimizer collapsed to angle ~0")

    print("\n" + "=" * 60)
    print("  Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
