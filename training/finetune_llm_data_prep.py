"""
Vortaz Labs — Data Preparation for Fine-Tuned LLM Baselines
=============================================================
Converts sciencebirds_data_v2.csv into instruction-tuning JSONL
for fair comparison: the LLMs see the SAME 50K trajectories the
JEPA was trained on.

Each row (state, action) → next_state becomes:
  instruction: "Given this game state... predict the next state after action..."
  response:    JSON with predicted next_state fields

If a trajectory resulted in a positive score, it ALSO generates an actor task:
  instruction: "You are an expert Angry Birds player... Choose the optimal shot."
  response:    JSON with the exact angle, power, tap_time that scored the points.

Output: data/finetune_train.jsonl, data/finetune_val.jsonl

Usage: python training/finetune_llm_data_prep.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# State vector layout (164D)
BIRD_FEATURES = 4    # type, x, y, status
MAX_BIRDS = 5
BLOCK_FEATURES = 6   # type, material, x, y, rotation, health
MAX_BLOCKS = 20
PIG_FEATURES = 4     # size, x, y, health
MAX_PIGS = 5


def state_to_text(state_vec):
    """Convert 164D state vector to readable structured text."""
    lines = []

    # Birds (0-19)
    birds = []
    for i in range(MAX_BIRDS):
        off = i * BIRD_FEATURES
        btype, bx, by, bstatus = state_vec[off:off+4]
        if abs(bx) > 0.001 or abs(by) > 0.001:
            birds.append(f"Bird{i+1}(type={btype:.2f},x={bx:.3f},y={by:.3f},used={bstatus:.0f})")
    if birds:
        lines.append("Birds: " + ", ".join(birds))

    # Blocks (20-139)
    blocks = []
    for i in range(MAX_BLOCKS):
        off = 20 + i * BLOCK_FEATURES
        btype, bmat, bx, by, brot, bhealth = state_vec[off:off+6]
        if abs(bhealth) > 0.001:
            blocks.append(f"Block{i+1}(type={btype:.2f},mat={bmat:.2f},x={bx:.3f},y={by:.3f},hp={bhealth:.2f})")
    if blocks:
        lines.append(f"Blocks({len(blocks)}): " + ", ".join(blocks[:10]))
        if len(blocks) > 10:
            lines.append(f"  ...and {len(blocks)-10} more blocks")

    # Pigs (140-159)
    pigs = []
    for i in range(MAX_PIGS):
        off = 140 + i * PIG_FEATURES
        psize, px, py, phealth = state_vec[off:off+4]
        if abs(phealth) > 0.001:
            pigs.append(f"Pig{i+1}(size={psize:.2f},x={px:.3f},y={py:.3f},hp={phealth:.2f})")
    if pigs:
        lines.append("Pigs: " + ", ".join(pigs))

    # Slingshot (160-161)
    sx, sy = state_vec[160], state_vec[161]
    lines.append(f"Slingshot: x={sx:.3f}, y={sy:.3f}")

    # Global (162-163)
    score, remaining = state_vec[162], state_vec[163]
    lines.append(f"Score: {score:.3f}, BirdsLeft: {remaining:.3f}")

    return "\n".join(lines)


def next_state_to_json(next_state_vec, score_delta):
    """Convert next_state vector to compact JSON output."""
    result = {}

    # Key fields only — pigs and blocks health (most important for scoring)
    pigs_alive = 0
    for i in range(MAX_PIGS):
        off = 140 + i * PIG_FEATURES
        hp = next_state_vec[off + 3]
        if abs(hp) > 0.001:
            pigs_alive += 1
            result[f"pig{i+1}_hp"] = round(float(hp), 3)

    # Block damage summary
    blocks_alive = sum(1 for i in range(MAX_BLOCKS)
                       if abs(next_state_vec[20 + i*6 + 5]) > 0.001)

    result["pigs_alive"] = pigs_alive
    result["blocks_remaining"] = blocks_alive
    result["score_delta"] = round(float(score_delta), 1)
    result["new_score"] = round(float(next_state_vec[162]), 3)
    result["birds_left"] = round(float(next_state_vec[163]), 3)

    return json.dumps(result)


def build_instruction(state_text, action):
    """Build the instruction prompt."""
    angle = action[0]
    power = action[1]
    tap_time = action[2]

    return (
        f"Given the current game state and an action taken by a player, "
        f"predict what the game state will look like after the action is executed.\n\n"
        f"Current State:\n{state_text}\n\n"
        f"Action: angle={angle:.3f}, power={power:.3f}, tap_time={tap_time:.3f}\n\n"
        f"Predict the resulting state as JSON with: pigs_alive, blocks_remaining, "
        f"score_delta, new_score, birds_left, and individual pig health values."
    )


def build_action_instruction(state_text):
    """Build the instruction for acting (playing the game)."""
    return (
        f"You are an expert Angry Birds player. Analyze this state and choose the optimal shot.\n\n"
        f"State:\n{state_text}\n\n"
        f"Respond with ONLY JSON: {{\"angle\": <0-90>, \"power\": <0-100>, \"tap_time\": <0-3>}}"
    )


def action_to_json(action):
    """Convert action array to JSON expected by choose_action()"""
    return json.dumps({
        "angle": round(float(action[0]), 3),
        "power": round(float(action[1] * 100), 1),  # convert 0-1 to 0-100
        "tap_time": round(float(action[2]), 3)
    })


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — Fine-Tune Data Preparation")
    print("=" * 60)

    # Find best data file
    data_dir = Path(__file__).resolve().parent.parent / "data"
    v2_csv = data_dir / "sciencebirds_data_v2.csv"
    v1_csv = data_dir / "sciencebirds_data.csv"
    csv_file = v2_csv if v2_csv.exists() else v1_csv

    if not csv_file.exists():
        print(f"ERROR: No data file found at {v2_csv} or {v1_csv}")
        print("Run data/collector.py first.")
        sys.exit(1)

    print(f"\n  Loading: {csv_file.name}")
    df = pd.read_csv(csv_file)
    print(f"  Rows: {len(df):,}")

    # Extract columns
    state_cols = [f"s_{i}" for i in range(164)]
    action_cols = [f"a_{i}" for i in range(3)]
    next_state_cols = [f"ns_{i}" for i in range(164)]

    states = df[state_cols].values.astype(np.float32)
    actions = df[action_cols].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    scores = df["score_delta"].values.astype(np.float32)

    # Build JSONL
    print(f"\n  Converting to instruction-tuning format...")

    records = []
    actor_tasks_added = 0

    for i in tqdm(range(len(df)), desc="  Converting rows", unit="row",
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        state_text = state_to_text(states[i])
        
        # 1. World Model Task (Predict Outcome)
        instruction = build_instruction(state_text, actions[i])
        response = next_state_to_json(next_states[i], scores[i])
        records.append({
            "instruction": instruction,
            "input": "",
            "output": response,
        })

        # 2. Actor Task (Behavioral Cloning of good shots)
        if scores[i] > 1000:  # Only clone shots that actually did decent damage
            actor_inst = build_action_instruction(state_text)
            actor_resp = action_to_json(actions[i])
            records.append({
                "instruction": actor_inst,
                "input": "",
                "output": actor_resp,
            })
            actor_tasks_added += 1

    # Split train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split = int(0.9 * len(records))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Write JSONL files
    train_path = data_dir / "finetune_train.jsonl"
    val_path = data_dir / "finetune_val.jsonl"

    print(f"\n  Writing train set: {len(train_idx):,} samples → {train_path.name}")
    with open(train_path, "w", encoding="utf-8") as f:
        for idx in tqdm(train_idx, desc="  Writing train", unit="row",
                        bar_format='{l_bar}{bar:30}{r_bar}'):
            f.write(json.dumps(records[idx]) + "\n")

    print(f"  Writing val set:   {len(val_idx):,} samples → {val_path.name}")
    with open(val_path, "w", encoding="utf-8") as f:
        for idx in tqdm(val_idx, desc="  Writing val  ", unit="row",
                        bar_format='{l_bar}{bar:30}{r_bar}'):
            f.write(json.dumps(records[idx]) + "\n")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Total records: {len(records):,}")

    # Show sample
    print(f"\n  --- Sample instruction (first record) ---")
    sample = records[0]
    print(f"  Instruction: {sample['instruction'][:200]}...")
    print(f"  Output: {sample['output'][:200]}...")


if __name__ == "__main__":
    main()
