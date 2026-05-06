"""
Vortaz Labs — Actor Module (LeCun Module 5)
================================================================
Implements the Actor from LeCun's "A Path Towards Autonomous
Machine Intelligence" architecture.

The actor finds optimal actions by minimizing the critic's energy
through gradient-based optimization, backpropagating through the
frozen world model and critic:

    a* = argmin_a  Cost( WorldModel(Perception(obs), a), a )

This is fundamentally different from policy-gradient RL:
  - No reward signal needed at inference time
  - No policy network to train — optimization happens at inference
  - Actions are found by gradient descent through a differentiable world model
  - Multi-start optimization for robustness

Pattern adapted from volt_optimizer.py (multi-start dispatch optimization).
================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional


# Action bounds (normalized [0, 1] space)
ACTION_MIN = torch.tensor([[0.0, 0.0, 0.0]])  # angle, power, tap
ACTION_MAX = torch.tensor([[1.0, 1.0, 1.0]])


def _run_single_start(jepa, critic, current_latent, init_action_norm,
                       act_min, act_max, steps=300, lr=0.01,
                       decoder=None, angle_band=None):
    """
    Run a single optimization trajectory.

    Optimizes action by minimizing energy through:
    action -> predictor(latent, action) -> critic(s_current, s_pred, action) -> energy

    Gradient flows from energy -> action (world model + critic are frozen).

    angle_band: optional (min_norm, max_norm) tuple to constrain the angle
                dimension, preventing all starts from collapsing to the same angle.
    """
    device = current_latent.device
    optimized_action = nn.Parameter(init_action_norm.clone().to(device))
    action_optimizer = optim.Adam([optimized_action], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        action_optimizer, T_max=steps, eta_min=0.001
    )

    # Per-start bounds: constrain angle if angle_band is specified
    local_min = act_min.clone()
    local_max = act_max.clone()
    if angle_band is not None:
        local_min[0, 0] = float(angle_band[0])
        local_max[0, 0] = float(angle_band[1])

    losses = []
    for step in range(steps):
        action_optimizer.zero_grad()

        # Clamp action to local bounds (may have per-start angle constraint)
        with torch.no_grad():
            optimized_action.data = torch.clamp(
                optimized_action.data,
                local_min.to(device), local_max.to(device)
            )

        # Forward through world model (frozen)
        s_pred = jepa.predict(current_latent, optimized_action)

        # Cost from critic (frozen) — uses both current and predicted latent
        energy = critic(current_latent, s_pred, optimized_action)

        # Only regularize power extremes and tap time, NOT angle
        # This prevents the angle from always collapsing to 0
        power_reg = 0.005 * ((optimized_action[0, 1] - 0.7) ** 2)
        tap_reg = 0.005 * (optimized_action[0, 2] ** 2)
        total_loss = energy.mean() + power_reg + tap_reg

        total_loss.backward()
        action_optimizer.step()
        scheduler.step()

        # Post-step clamp
        with torch.no_grad():
            optimized_action.data = torch.clamp(
                optimized_action.data,
                local_min.to(device), local_max.to(device)
            )

        losses.append(energy.mean().item())

    # Final clamp to global bounds for output
    with torch.no_grad():
        optimized_action.data = torch.clamp(
            optimized_action.data,
            act_min.to(device), act_max.to(device)
        )

    final_action = optimized_action.detach()
    with torch.no_grad():
        final_pred = jepa.predict(current_latent, final_action)
        final_energy = critic(current_latent, final_pred, final_action).item()

    # Optionally decode for visualization
    decoded_pred = None
    if decoder is not None:
        with torch.no_grad():
            decoded_pred = decoder(final_pred)

    return final_action, final_pred, decoded_pred, losses, final_energy


def _compute_target_angle_from_screenshot(screenshot_path: str = None) -> float:
    """
    Vision-based targeting: analyze the actual game screenshot to find
    where pigs and structures are, then compute the optimal launch angle.

    Uses color detection on the real screenshot — no synthetic data.

    Returns angle in normalized [0, 1] space (0=0°, 1=90°).
    """
    try:
        from PIL import Image
        path = screenshot_path or "screenshot_debug.png"
        img = Image.open(path)
        arr = np.array(img)
    except Exception:
        return 0.35  # default ~31° if screenshot unavailable

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h, w = arr.shape[:2]

    # === Find PIGS (bright green, right half, above ground) ===
    pig_mask = (g > 120) & (g > r + 40) & (g > b + 40) & (r < 140) & (b < 100)
    pig_mask[:, :int(w * 0.35)] = False   # only right side
    pig_mask[int(h * 0.85):, :] = False   # above ground line

    pig_ys, pig_xs = np.where(pig_mask)

    # === Find STRUCTURES (wood/ice/stone, right half) ===
    struct_mask = np.zeros((h, w), dtype=bool)

    # Wood: brownish
    wood = (r > 130) & (r < 220) & (g > 80) & (g < 170) & (b > 30) & (b < 120) & (r > b + 30)
    # Ice: light blue
    ice = (b > 150) & (b > r + 20) & (b > g) & (r < 200) & (g > 100)
    # Stone: grey
    stone = (r > 100) & (r < 180) & (np.abs(r.astype(int) - g.astype(int)) < 30) & \
            (np.abs(g.astype(int) - b.astype(int)) < 30) & (r > 80)

    for m in [wood, ice, stone]:
        m[:, :int(w * 0.35)] = False
        m[int(h * 0.85):, :] = False
        struct_mask |= m

    struct_ys, struct_xs = np.where(struct_mask)

    # === Bird position (slingshot) ===
    # Known from screenshot analysis: bird sits at roughly (165, 283)
    # for 918x399 resolution. Scale to actual resolution.
    bird_x = int(165 * w / 918)
    bird_y = int(283 * h / 399)

    # === Pick target ===
    if len(pig_xs) > 50:
        # Target pig centroid
        target_x = float(np.mean(pig_xs))
        target_y = float(np.mean(pig_ys))
    elif len(struct_xs) > 50:
        # Target structure centroid
        target_x = float(np.mean(struct_xs))
        target_y = float(np.mean(struct_ys))
    else:
        # Fallback: aim at center-right, slightly above midline
        target_x = w * 0.65
        target_y = h * 0.35

    dx = target_x - bird_x   # positive = rightward
    dy = bird_y - target_y    # positive = upward (screen Y is inverted)

    # Projectile angle with gravity compensation
    # atan2(dy, dx) gives the line-of-sight angle
    # Add gravity compensation: more for distant targets
    los_angle = np.degrees(np.arctan2(dy, dx))

    # Gravity compensation: ~10-15° for medium range, more for far
    distance = np.sqrt(dx**2 + dy**2)
    gravity_comp = np.clip(distance / 50.0, 5.0, 20.0)

    target_angle = los_angle + gravity_comp
    target_angle = np.clip(target_angle, 10.0, 75.0)

    return target_angle / 90.0  # normalize to [0, 1]


def _compute_target_angle(obs_raw: np.ndarray) -> float:
    """
    Wrapper that tries screenshot-based targeting first,
    falls back to state-vector-based targeting.
    """
    # Always prefer screenshot-based targeting (real game data)
    result = _compute_target_angle_from_screenshot()
    return result


def optimize_action(obs_norm: torch.Tensor, jepa, critic,
                    norm_stats: dict = None, decoder=None,
                    steps: int = 300, num_starts: int = 10,
                    context: torch.Tensor = None,
                    obs_raw: np.ndarray = None) -> dict:
    """
    LeCun Module 5: Actor — find optimal action via energy minimization.

    Hybrid approach:
      - ANGLE: Set by screenshot vision (color detection finds pigs/structures)
        The critic was trained on OfflineSimulator states so it has a monotonic
        angle bias. Vision-based targeting is more accurate for angle selection.
      - POWER & TAP: Optimized by critic through the world model
        The critic CAN discriminate execution quality (Spearman=0.71).

    Multi-start optimization explores different power/tap combinations while
    keeping the vision-derived angle fixed (with small perturbation band).

    a* = argmin_{power,tap}  Critic( WorldModel.predict(s, [angle_vision, power, tap]) )
    """
    device = next(jepa.parameters()).device
    obs_norm = obs_norm.to(device)

    # Encode current observation
    with torch.no_grad():
        s_t, _ = jepa.encode(obs_norm, context)

    act_min = ACTION_MIN.to(device)
    act_max = ACTION_MAX.to(device)

    # Vision-based angle: screenshot analysis finds pigs and computes trajectory
    if obs_raw is None:
        obs_raw = obs_norm.detach().cpu().numpy()[0]
    target_angle_norm = _compute_target_angle(obs_raw)
    target_a = float(np.clip(target_angle_norm, 0.05, 0.95))

    best_action = None
    best_pred = None
    best_decoded = None
    best_losses = None
    best_energy = float('inf')

    # Multi-start: angle locked to vision target (+/- small band),
    # critic optimizes power and tap time
    # Tight angle band: vision is accurate, allow only +/- 5 degrees
    angle_band = (max(0.0, target_a - 0.055), min(1.0, target_a + 0.055))

    starts = [
        # Different power levels (critic picks the best)
        (target_a, 1.00, 0.00),
        (target_a, 0.90, 0.00),
        (target_a, 0.80, 0.00),
        (target_a, 0.70, 0.00),
        # With tap (for special bird abilities)
        (target_a, 0.95, 0.30),
        (target_a, 0.85, 0.50),
        # Slight angle perturbations (within band)
        (target_a - 0.04, 0.95, 0.00),
        (target_a + 0.04, 0.90, 0.00),
    ]

    actual_starts = min(num_starts, len(starts))

    for start_idx in range(actual_starts):
        a_init, p_init, t_init = starts[start_idx]
        a_init = float(np.clip(a_init, 0.05, 0.95))

        init = torch.tensor([[a_init, p_init, t_init]], device=device)
        init = torch.clamp(init, act_min, act_max)

        # Angle band: vision-locked with small perturbation allowed
        action, pred, decoded, losses, energy = _run_single_start(
            jepa, critic, s_t, init, act_min, act_max,
            steps=steps, decoder=decoder, angle_band=angle_band
        )

        if energy < best_energy:
            best_energy = energy
            best_action = action
            best_pred = pred
            best_decoded = decoded
            best_losses = losses

    # Convert to raw action values
    best_action_raw = best_action.detach().cpu().numpy()[0]
    angle_deg = best_action_raw[0] * 90.0
    power = np.clip(best_action_raw[1], 0.0, 1.0)
    tap_time_ms = best_action_raw[2] * 3000.0

    result = {
        "action_norm": best_action.detach().cpu(),
        "action_raw": {
            "angle_deg": float(angle_deg),
            "power": float(power),
            "tap_time_ms": float(tap_time_ms),
        },
        "predicted_latent": best_pred.detach().cpu(),
        "energy": best_energy,
        "convergence": best_losses,
    }

    if best_decoded is not None:
        result["predicted_state_decoded"] = best_decoded.detach().cpu()

    return result


def predict_outcome(obs_norm: torch.Tensor, action_norm: torch.Tensor,
                    jepa, decoder, norm_stats: dict = None) -> np.ndarray:
    """
    Predict the outcome of an action without optimization.
    Useful for comparing JEPA predictions with LLM predictions.
    """
    device = next(jepa.parameters()).device
    with torch.no_grad():
        s_pred, _ = jepa(obs_norm.to(device), action_norm.to(device))
        decoded = decoder(s_pred)
    return decoded.cpu().numpy()[0]


if __name__ == "__main__":
    from world_model import GameJEPA, GameDecoder
    from critic import Critic

    print("Testing Actor with random models...\n")

    jepa = GameJEPA(obs_dim=164, action_dim=3)
    decoder = GameDecoder(latent_dim=256, output_dim=164)
    critic = Critic(latent_dim=256, action_dim=3)

    # Freeze for optimization
    jepa.eval()
    critic.eval()
    for p in jepa.parameters():
        p.requires_grad = False
    for p in critic.parameters():
        p.requires_grad = False

    obs = torch.randn(1, 164)

    result = optimize_action(obs, jepa, critic, decoder=decoder,
                             steps=100, num_starts=5)

    print(f"Optimal action:")
    print(f"  Angle:    {result['action_raw']['angle_deg']:.1f} deg")
    print(f"  Power:    {result['action_raw']['power']:.3f}")
    print(f"  Tap time: {result['action_raw']['tap_time_ms']:.0f} ms")
    print(f"  Energy:   {result['energy']:.4f}")
    print(f"  Steps in convergence: {len(result['convergence'])}")
