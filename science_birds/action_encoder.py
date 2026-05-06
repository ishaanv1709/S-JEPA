"""
Action Encoder — Converts between different action representations.

Supports:
  - Polar (angle_deg, power, tap_time) — intuitive for humans/LLMs
  - Cartesian (dx, dy, tap_time) — what Science Birds socket API expects
  - Normalized (all in [0,1]) — what the JEPA model uses
"""

import numpy as np


# Action bounds (raw)
ANGLE_MIN, ANGLE_MAX = 0.0, 90.0
POWER_MIN, POWER_MAX = 0.0, 1.0
TAP_TIME_MIN, TAP_TIME_MAX = 0.0, 3000  # milliseconds

# Slingshot pull distance (pixels)
MAX_PULL_DISTANCE = 80.0


def polar_to_cartesian(angle_deg: float, power: float) -> tuple:
    """
    Convert angle/power to dx/dy pull offsets for the slingshot.

    The pull is in the opposite direction of the launch.
    """
    pull_distance = power * MAX_PULL_DISTANCE
    angle_rad = np.radians(angle_deg)
    dx = -pull_distance * np.cos(angle_rad)
    dy = pull_distance * np.sin(angle_rad)
    return float(dx), float(dy)


def cartesian_to_polar(dx: float, dy: float) -> tuple:
    """Convert dx/dy pull offsets back to angle/power."""
    pull_distance = np.sqrt(dx**2 + dy**2)
    power = pull_distance / MAX_PULL_DISTANCE

    if pull_distance < 1e-6:
        return 0.0, 0.0

    angle_rad = np.arctan2(dy, -dx)
    angle_deg = np.degrees(angle_rad)
    return float(np.clip(angle_deg, 0, 90)), float(np.clip(power, 0, 1))


def normalize_action(angle_deg: float, power: float,
                     tap_time_ms: float) -> np.ndarray:
    """Normalize raw action to [0, 1] range for model input."""
    return np.array([
        (angle_deg - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN),
        (power - POWER_MIN) / (POWER_MAX - POWER_MIN),
        (tap_time_ms - TAP_TIME_MIN) / (TAP_TIME_MAX - TAP_TIME_MIN),
    ], dtype=np.float32)


def denormalize_action(action_norm: np.ndarray) -> tuple:
    """Convert normalized action back to raw values."""
    angle = action_norm[0] * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
    power = action_norm[1] * (POWER_MAX - POWER_MIN) + POWER_MIN
    tap_ms = action_norm[2] * (TAP_TIME_MAX - TAP_TIME_MIN) + TAP_TIME_MIN
    return float(angle), float(np.clip(power, 0, 1)), float(tap_ms)


def clamp_action(action_norm: np.ndarray) -> np.ndarray:
    """Clamp normalized action to valid bounds [0, 1]."""
    return np.clip(action_norm, 0.0, 1.0)


def sample_random_action(rng: np.random.RandomState = None) -> tuple:
    """Sample a random action in raw (angle, power, tap_time_ms) format."""
    if rng is None:
        rng = np.random.RandomState()
    angle = rng.uniform(10, 80)       # avoid extreme angles
    power = rng.uniform(0.3, 1.0)     # minimum useful power
    tap_time = rng.choice([0, 0, 0, 500, 1000, 1500])  # mostly no tap
    return angle, power, float(tap_time)


def sample_heuristic_action(target_x: float, target_y: float,
                            slingshot_x: float = 100.0,
                            slingshot_y: float = 200.0,
                            rng: np.random.RandomState = None) -> tuple:
    """
    Sample an action aimed at a target using proper projectile physics.
    Uses the ACTUAL OfflineSimulator constants (GRAVITY=9.81, MAX_LAUNCH_SPEED=50).
    Brute-force searches angle/power space to find shots that hit.
    """
    if rng is None:
        rng = np.random.RandomState()

    dx = target_x - slingshot_x
    dy = target_y - slingshot_y
    g = 9.81       # OfflineSimulator.GRAVITY
    max_v = 50.0   # OfflineSimulator.MAX_LAUNCH_SPEED
    dt = 0.02      # OfflineSimulator timestep

    best_angle = 35.0
    best_power = 0.8
    best_miss = float('inf')

    # Vectorized search: simulate all angle/power combos at once
    angles = np.arange(5, 80, 5, dtype=np.float64)
    powers = np.array([0.7, 0.85, 1.0])

    # Create all combos
    aa, pp = np.meshgrid(angles, powers)
    aa = aa.ravel()
    pp = pp.ravel()
    n = len(aa)

    v = pp * max_v
    angle_rad = np.radians(aa)
    vx = v * np.cos(angle_rad)
    vy = v * np.sin(angle_rad)
    px = np.full(n, slingshot_x)
    py = np.full(n, slingshot_y)
    best_misses = np.full(n, 1e9)

    for _ in range(400):
        vy -= g * dt
        px += vx * dt
        py += vy * dt
        miss = np.sqrt((px - target_x)**2 + (py - target_y)**2)
        better = miss < best_misses
        best_misses[better] = miss[better]

    idx = np.argmin(best_misses)
    best_miss = best_misses[idx]
    best_angle = float(aa[idx])
    best_power = float(pp[idx])

    # Add controlled noise for data diversity
    angle = np.clip(best_angle + rng.normal(0, 3), 5, 85)
    power = np.clip(best_power + rng.normal(0, 0.06), 0.3, 1.0)
    tap_time = rng.choice([0, 0, 500, 1000])
    return float(angle), float(power), float(tap_time)
