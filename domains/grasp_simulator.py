"""
Vortaz Labs — 2D Robotic Grasping Simulator
=============================================
A physics-based 2D grasping environment for cross-domain transfer testing.
Different state space and dynamics from Science Birds, but the underlying
causal structure (force → motion → outcome) is analogous.

State (32D):
  Gripper: x, y, aperture, force_x, force_y, gripping (6)
  Objects (5): x, y, width, height, mass (5 * 5 = 25)
  Global: timestep (1)

Action (3D):
  dx, dy, d_aperture  (normalized [-1, 1])

Generates training data: (state, action, next_state, score)

Usage:
  python domains/grasp_simulator.py       # generate training data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

OBS_DIM = 32
ACTION_DIM = 3
MAX_OBJECTS = 5
OBJECT_FEATURES = 5  # x, y, width, height, mass

ARENA_W = 1.0
ARENA_H = 1.0
GRAVITY = -0.01
GRIPPER_SPEED = 0.05
APERTURE_SPEED = 0.1


class GraspSimulator:
    """Simple 2D physics for robotic grasping."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_scene(self, n_objects=None):
        """Generate a random grasping scene."""
        if n_objects is None:
            n_objects = self.rng.randint(1, MAX_OBJECTS + 1)

        objects = []
        for _ in range(n_objects):
            obj = {
                "x": self.rng.uniform(0.2, 0.8),
                "y": self.rng.uniform(0.05, 0.4),  # on ground/table
                "width": self.rng.uniform(0.03, 0.1),
                "height": self.rng.uniform(0.03, 0.1),
                "mass": self.rng.uniform(0.1, 1.0),
                "grasped": False,
                "velocity_y": 0.0,
            }
            objects.append(obj)

        gripper = {
            "x": self.rng.uniform(0.3, 0.7),
            "y": self.rng.uniform(0.6, 0.9),
            "aperture": 1.0,  # open = 1.0, closed = 0.0
            "force_x": 0.0,
            "force_y": 0.0,
            "gripping": 0.0,
        }

        return {"gripper": gripper, "objects": objects, "timestep": 0}

    def state_to_vector(self, scene):
        """Convert scene dict to flat 32D state vector."""
        g = scene["gripper"]
        state = np.zeros(OBS_DIM, dtype=np.float32)

        # Gripper (6D)
        state[0] = g["x"]
        state[1] = g["y"]
        state[2] = g["aperture"]
        state[3] = g["force_x"]
        state[4] = g["force_y"]
        state[5] = g["gripping"]

        # Objects (5 x 5D)
        for i, obj in enumerate(scene["objects"][:MAX_OBJECTS]):
            off = 6 + i * OBJECT_FEATURES
            state[off + 0] = obj["x"]
            state[off + 1] = obj["y"]
            state[off + 2] = obj["width"]
            state[off + 3] = obj["height"]
            state[off + 4] = obj["mass"]

        # Global
        state[31] = scene["timestep"] / 100.0  # normalized timestep

        return state

    def physics_step(self, scene, action):
        """Advance physics by one timestep. Returns (new_scene, score)."""
        dx, dy, d_ap = action[0], action[1], action[2]
        g = scene["gripper"].copy()
        objects = [obj.copy() for obj in scene["objects"]]
        step = scene["timestep"] + 1

        # Move gripper
        g["x"] = np.clip(g["x"] + dx * GRIPPER_SPEED, 0.0, 1.0)
        g["y"] = np.clip(g["y"] + dy * GRIPPER_SPEED, 0.0, 1.0)
        g["aperture"] = np.clip(g["aperture"] + d_ap * APERTURE_SPEED, 0.0, 1.0)

        # Track forces
        g["force_x"] = dx * GRIPPER_SPEED
        g["force_y"] = dy * GRIPPER_SPEED

        score = 0.0

        for obj in objects:
            # Check if gripper is near object
            near_x = abs(g["x"] - obj["x"]) < obj["width"] + 0.02
            near_y = abs(g["y"] - obj["y"]) < obj["height"] + 0.02

            if near_x and near_y and g["aperture"] < 0.3:
                # Gripper closed near object → grasp attempt
                if not obj["grasped"]:
                    grip_force = (1.0 - g["aperture"]) * 2.0
                    if grip_force > obj["mass"] * 0.5:
                        obj["grasped"] = True
                        g["gripping"] = 1.0
                        score += 10.0  # grasping reward
                        print(f"    [GRASP] Object at ({obj['x']:.2f}, {obj['y']:.2f}) grasped!") if step % 20 == 0 else None

            if obj["grasped"]:
                # Object follows gripper
                obj["x"] = g["x"]
                obj["y"] = g["y"] - obj["height"] / 2
                obj["velocity_y"] = 0.0

                # Lifting bonus
                if obj["y"] > 0.5:
                    score += 5.0

                # Release check
                if g["aperture"] > 0.7:
                    obj["grasped"] = False
                    g["gripping"] = 0.0

            elif obj["y"] > 0.01:
                # Object falls under gravity
                obj["velocity_y"] += GRAVITY
                obj["y"] = max(0.01, obj["y"] + obj["velocity_y"])

        return {"gripper": g, "objects": objects, "timestep": step}, score

    def sample_action(self, scene):
        """Sample a context-aware action."""
        g = scene["gripper"]

        # Find closest ungrasped object
        closest_obj = None
        min_dist = float('inf')
        for obj in scene["objects"]:
            if not obj["grasped"]:
                dist = np.sqrt((g["x"] - obj["x"])**2 + (g["y"] - obj["y"])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = obj

        if closest_obj is not None and self.rng.random() < 0.7:
            # Move toward closest object
            dx = np.sign(closest_obj["x"] - g["x"]) + self.rng.normal(0, 0.3)
            dy = np.sign(closest_obj["y"] - g["y"]) + self.rng.normal(0, 0.3)

            # Close gripper when near
            if min_dist < 0.1:
                d_ap = -1.0 + self.rng.normal(0, 0.2)  # close
            else:
                d_ap = self.rng.normal(0, 0.3)
        else:
            # Random exploration
            dx = self.rng.uniform(-1, 1)
            dy = self.rng.uniform(-1, 1)
            d_ap = self.rng.uniform(-1, 1)

        return np.clip([dx, dy, d_ap], -1.0, 1.0)


def generate_grasping_data(n_episodes=1000, episode_length=50, output_dir="data"):
    """Generate training data for the grasping domain."""
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — Grasping Domain Data Generation")
    print("=" * 60)

    records = []
    total_grasps = 0

    for ep in tqdm(range(n_episodes), desc="  Episodes", unit="ep",
                    bar_format='{l_bar}{bar:30}{r_bar}'):
        sim = GraspSimulator(seed=ep)
        scene = sim.generate_scene()

        for t in range(episode_length):
            state = sim.state_to_vector(scene)
            action = sim.sample_action(scene)
            next_scene, score = sim.physics_step(scene, action)
            next_state = sim.state_to_vector(next_scene)

            records.append({
                **{f"s_{i}": state[i] for i in range(OBS_DIM)},
                **{f"a_{i}": action[i] for i in range(ACTION_DIM)},
                **{f"ns_{i}": next_state[i] for i in range(OBS_DIM)},
                "score_delta": score,
                "episode": ep,
                "step": t,
            })

            if score > 0:
                total_grasps += 1
            scene = next_scene

    df = pd.DataFrame(records)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "grasping_data.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Saved {len(df):,} rows to {out_path}")
    print(f"  Total grasps: {total_grasps}")
    print(f"  Grasp rate: {100*total_grasps/len(df):.1f}%")

    return str(out_path)


if __name__ == "__main__":
    generate_grasping_data(n_episodes=1000, episode_length=50)
