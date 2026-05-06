"""
State Parser — Converts Science Birds game state into fixed-size NumPy arrays
for the JEPA world model.

State vector layout (164 dimensions):
  Birds:     5 birds  x 4 features = 20  (type_idx, x, y, used)
  Blocks:    20 blocks x 6 features = 120 (type, material, x, y, rotation, health_pct)
  Pigs:      5 pigs   x 4 features = 20  (size_idx, x, y, health_pct)
  Slingshot: 2 features                  (x, y)
  Global:    2 features                  (score_normalized, birds_remaining)
  Total: 164 dimensions
"""

import numpy as np
from typing import Optional


# Feature indices
MAX_BIRDS = 5
MAX_BLOCKS = 20
MAX_PIGS = 5
BIRD_FEATURES = 4
BLOCK_FEATURES = 6
PIG_FEATURES = 4
SLINGSHOT_FEATURES = 2
GLOBAL_FEATURES = 2

OBS_DIM = (MAX_BIRDS * BIRD_FEATURES +
           MAX_BLOCKS * BLOCK_FEATURES +
           MAX_PIGS * PIG_FEATURES +
           SLINGSHOT_FEATURES +
           GLOBAL_FEATURES)  # = 164

# Encoding maps
BIRD_TYPE_MAP = {"red": 0, "blue": 1, "yellow": 2, "black": 3, "white": 4}
MATERIAL_MAP = {"wood": 0, "ice": 1, "stone": 2}
BLOCK_TYPE_MAP = {"rect": 0, "square": 1, "circle": 2, "triangle": 3}
PIG_SIZE_MAP = {"small": 0, "medium": 1, "large": 2}

# Normalization constants (approximate Science Birds coordinate ranges)
NORM_X = 800.0    # arena width
NORM_Y = 400.0    # arena height
NORM_ROT = 360.0  # rotation degrees
NORM_SCORE = 50000.0  # typical max score


def parse_state(ground_truth: dict) -> np.ndarray:
    """
    Convert a Science Birds ground truth dict into a fixed-size state vector.

    Args:
        ground_truth: dict with keys 'birds', 'blocks', 'pigs', 'slingshot', 'score'

    Returns:
        np.ndarray of shape (164,) — normalized state vector
    """
    state = np.zeros(OBS_DIM, dtype=np.float32)
    offset = 0

    # === Birds (5 x 4) ===
    birds = ground_truth.get("birds", [])
    for i in range(MAX_BIRDS):
        if i < len(birds):
            bird = birds[i]
            state[offset + 0] = BIRD_TYPE_MAP.get(bird.get("type", "red"), 0) / 4.0
            state[offset + 1] = bird.get("x", 0) / NORM_X
            state[offset + 2] = bird.get("y", 0) / NORM_Y
            state[offset + 3] = 1.0 if bird.get("used", False) else 0.0
        # else: zeros (padding for absent birds)
        offset += BIRD_FEATURES

    # === Blocks (20 x 6) ===
    blocks = ground_truth.get("blocks", [])
    for i in range(MAX_BLOCKS):
        if i < len(blocks):
            block = blocks[i]
            state[offset + 0] = BLOCK_TYPE_MAP.get(block.get("type", "rect"), 0) / 3.0
            state[offset + 1] = MATERIAL_MAP.get(block.get("material", "wood"), 0) / 2.0
            state[offset + 2] = block.get("x", 0) / NORM_X
            state[offset + 3] = block.get("y", 0) / NORM_Y
            state[offset + 4] = block.get("rotation", 0) / NORM_ROT
            # Health as percentage of max
            max_hp = block.get("max_health", 100.0)
            hp = block.get("health", max_hp)
            state[offset + 5] = hp / max_hp if max_hp > 0 else 0.0
        offset += BLOCK_FEATURES

    # === Pigs (5 x 4) ===
    pigs = ground_truth.get("pigs", [])
    for i in range(MAX_PIGS):
        if i < len(pigs):
            pig = pigs[i]
            state[offset + 0] = PIG_SIZE_MAP.get(pig.get("size", "medium"), 1) / 2.0
            state[offset + 1] = pig.get("x", 0) / NORM_X
            state[offset + 2] = pig.get("y", 0) / NORM_Y
            max_hp = pig.get("max_health", 100.0)
            hp = pig.get("health", max_hp)
            state[offset + 3] = hp / max_hp if max_hp > 0 else 0.0
        offset += PIG_FEATURES

    # === Slingshot (2) ===
    slingshot = ground_truth.get("slingshot", {})
    state[offset + 0] = slingshot.get("x", 100.0) / NORM_X
    state[offset + 1] = slingshot.get("y", 200.0) / NORM_Y
    offset += SLINGSHOT_FEATURES

    # === Global (2) ===
    state[offset + 0] = ground_truth.get("score", 0) / NORM_SCORE
    birds_remaining = sum(1 for b in birds if not b.get("used", False))
    state[offset + 1] = birds_remaining / MAX_BIRDS
    offset += GLOBAL_FEATURES

    return state


def parse_action(angle_deg: float, power: float, tap_time: float) -> np.ndarray:
    """
    Normalize action to [0, 1] range.

    Args:
        angle_deg: launch angle (0-90 degrees)
        power: launch power (0-1)
        tap_time: tap time in seconds (0-3)

    Returns:
        np.ndarray of shape (3,) — normalized action
    """
    return np.array([
        angle_deg / 90.0,
        power,
        tap_time / 3.0,
    ], dtype=np.float32)


def unparse_action(action_norm: np.ndarray) -> tuple:
    """
    Convert normalized action back to raw values.

    Returns:
        (angle_deg, power, tap_time)
    """
    return (
        float(action_norm[0] * 90.0),
        float(np.clip(action_norm[1], 0.0, 1.0)),
        float(action_norm[2] * 3.0),
    )


def state_to_description(ground_truth: dict) -> str:
    """
    Convert game state to human-readable text for LLM prompting.

    Returns a structured text description of the current game state.
    """
    lines = []

    # Birds
    birds = ground_truth.get("birds", [])
    available = [b for b in birds if not b.get("used", False)]
    lines.append(f"Birds available: {len(available)}")
    if available:
        current = available[0]
        lines.append(f"  Current bird: {current['type'].capitalize()} "
                     f"(at slingshot)")
        for b in available[1:]:
            lines.append(f"  Waiting: {b['type'].capitalize()}")

    # Slingshot
    sling = ground_truth.get("slingshot", {})
    lines.append(f"Slingshot position: ({sling.get('x', 100):.0f}, "
                 f"{sling.get('y', 200):.0f})")

    # Blocks
    blocks = ground_truth.get("blocks", [])
    alive_blocks = [b for b in blocks if b.get("health", 0) > 0]
    lines.append(f"\nBlocks: {len(alive_blocks)} remaining")
    for i, block in enumerate(alive_blocks[:10]):  # limit for prompt length
        hp_pct = block["health"] / block.get("max_health", 100) * 100
        lines.append(f"  [{i}] {block['material'].capitalize()} "
                     f"at ({block['x']:.0f}, {block['y']:.0f}) "
                     f"rot={block.get('rotation', 0):.0f}deg "
                     f"HP={hp_pct:.0f}%")
    if len(alive_blocks) > 10:
        lines.append(f"  ... and {len(alive_blocks) - 10} more blocks")

    # Pigs
    pigs = ground_truth.get("pigs", [])
    alive_pigs = [p for p in pigs if p.get("health", 0) > 0]
    lines.append(f"\nPigs: {len(alive_pigs)} alive")
    for i, pig in enumerate(alive_pigs):
        hp_pct = pig["health"] / pig.get("max_health", 100) * 100
        lines.append(f"  [{i}] {pig.get('size', 'medium').capitalize()} pig "
                     f"at ({pig['x']:.0f}, {pig['y']:.0f}) HP={hp_pct:.0f}%")

    # Score
    lines.append(f"\nScore: {ground_truth.get('score', 0)}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample level
    from client import OfflineSimulator

    sim = OfflineSimulator(seed=42)
    level = sim.generate_level("medium")

    state_vec = parse_state(level)
    print(f"State vector shape: {state_vec.shape}")
    print(f"State vector range: [{state_vec.min():.3f}, {state_vec.max():.3f}]")
    print(f"Non-zero elements: {np.count_nonzero(state_vec)} / {len(state_vec)}")

    action = parse_action(45.0, 0.8, 1.0)
    print(f"\nAction vector: {action}")
    print(f"Unparsed: {unparse_action(action)}")

    desc = state_to_description(level)
    print(f"\nText description:\n{desc}")
