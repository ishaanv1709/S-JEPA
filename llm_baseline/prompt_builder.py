"""
Prompt Builder — Converts game state to structured text prompts for LLMs.

Designs prompts that give LLMs the same information the JEPA world model
receives, ensuring a fair comparison. The LLM must reason about physics
from text descriptions while the JEPA reasons from learned latent dynamics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from science_birds.state_parser import state_to_description


SYSTEM_PROMPT = """You are an expert Angry Birds player with deep understanding of projectile physics.
You must analyze the game state and choose the optimal shot to maximize damage to pigs and structures.

Physics rules:
- Projectiles follow parabolic trajectories under gravity (9.81 m/s^2)
- Higher angles give more height but less distance
- Higher power gives more speed and more impact force
- Impact damage = mass * speed * coefficient
- Materials have different strengths: ice (weakest) < wood < stone (strongest)
- Destroying blocks above other blocks causes them to fall (chain damage)
- Bird abilities activate at tap_time seconds after launch

Bird abilities:
- Red: No special ability (standard impact)
- Blue: Splits into 3 smaller birds (good for ice)
- Yellow: Speed boost on tap (good for wood, penetrating)
- Black: Explodes on tap (area damage, good for stone)
- White: Drops egg bomb on tap (downward damage)

You must respond with ONLY a JSON object, no other text:
{"angle": <float 0-90>, "power": <float 0-100>, "tap_time": <float 0-3>}"""


def build_action_prompt(ground_truth: dict) -> list:
    """
    Build a chat prompt for the LLM to choose an action.

    Returns list of message dicts for the chat API.
    """
    state_desc = state_to_description(ground_truth)

    user_msg = f"""Current game state:
{state_desc}

Analyze the positions of pigs relative to blocks. Consider:
1. Which pig is easiest to hit?
2. What trajectory (angle + power) will reach it?
3. Can you cause chain reactions by hitting support blocks?
4. Should you use the bird's special ability?

Choose the optimal shot. Respond with ONLY JSON:
{{"angle": <0-90 degrees>, "power": <0-100 percent>, "tap_time": <0-3 seconds>}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_prediction_prompt(ground_truth: dict,
                            angle: float, power: float,
                            tap_time: float) -> list:
    """
    Build a prompt asking the LLM to PREDICT the outcome of a shot.
    This tests the LLM's world modeling ability directly.

    Used for prediction accuracy comparison with JEPA.
    """
    state_desc = state_to_description(ground_truth)

    user_msg = f"""Current game state:
{state_desc}

A shot is about to be taken with these parameters:
- Launch angle: {angle:.1f} degrees
- Launch power: {power:.1f}%
- Tap time: {tap_time:.2f} seconds

Predict the outcome. Consider the projectile trajectory, which blocks/pigs
it will hit, and the resulting damage. Respond with ONLY JSON:
{{
  "predicted_score_delta": <int, estimated points gained>,
  "pigs_killed": <int, number of pigs destroyed>,
  "blocks_destroyed": <int, approximate blocks broken>,
  "projectile_landing_x": <float, where the bird approximately lands>,
  "reasoning": "<brief 1-sentence explanation of your prediction>"
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


if __name__ == "__main__":
    from science_birds.client import OfflineSimulator

    sim = OfflineSimulator(seed=42)
    level = sim.generate_level("medium")

    # Action prompt
    messages = build_action_prompt(level)
    print("=== ACTION PROMPT ===")
    for msg in messages:
        print(f"\n[{msg['role']}]:")
        print(msg['content'][:500])

    # Prediction prompt
    print("\n\n=== PREDICTION PROMPT ===")
    pred_messages = build_prediction_prompt(level, 45.0, 80.0, 1.0)
    for msg in pred_messages:
        print(f"\n[{msg['role']}]:")
        print(msg['content'][:500])
