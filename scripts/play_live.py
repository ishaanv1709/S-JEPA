"""
Vortaz Labs — Watch Your World Model Play Angry Birds LIVE
===========================================================
This script connects to the actual Science Birds Unity game
and plays it using the trained LeWorldModel in real-time.

SETUP:
  1. Open Unity Hub
  2. Add project: C:/Users/ishaa_04bpft8/Energy Grid World Model/science-birds
  3. Open it in Unity 2019.4+ (or any compatible version)
  4. Press PLAY in Unity — the WebSocket server starts at ws://localhost:9000
  5. Run this script: python play_live.py

You'll see the world model's shots execute in real-time in Unity!
"""

import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import ScienceBirdsClient
from science_birds.state_parser import parse_state, OBS_DIM
from models.world_model import GameJEPA, GameDecoder
from models.critic import Critic
from models.actor import optimize_action
from data.dataset import GameDataset


def load_models(device="cpu"):
    """Load trained world model components."""
    print("Loading trained LeWorldModel...")

    # Prefer v2 checkpoint (better critic, more data), fallback to latest epoch
    ckpt_dir = Path("checkpoints")
    v2_ckpt = ckpt_dir / "game_jepa_best_v2.pth"
    if v2_ckpt.exists():
        best_ckpt = str(v2_ckpt)
    else:
        checkpoints = sorted(ckpt_dir.glob("game_jepa_ep*.pth"))
        if not checkpoints:
            print("ERROR: No JEPA checkpoints found. Train the model first.")
            sys.exit(1)
        best_ckpt = str(checkpoints[-1])
    print(f"  JEPA: {best_ckpt}")

    jepa = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                    hidden_dim=512, use_memory=False,
                    use_configurator=False).to(device)
    jepa.load_state_dict(
        torch.load(best_ckpt, map_location=device)['model_state_dict']
    )
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    decoder = GameDecoder(latent_dim=256, output_dim=164).to(device)
    dec_path = "checkpoints/game_decoder_v2.pth" if Path("checkpoints/game_decoder_v2.pth").exists() else "checkpoints/game_decoder.pth"
    if Path(dec_path).exists():
        decoder.load_state_dict(
            torch.load(dec_path, map_location=device)
        )
        print(f"  Decoder: {dec_path}")
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    critic = Critic(latent_dim=256, action_dim=3, hidden_dim=256).to(device)
    crit_path = "checkpoints/game_critic_v2.pth" if Path("checkpoints/game_critic_v2.pth").exists() else "checkpoints/game_critic.pth"
    if Path(crit_path).exists():
        critic.load_state_dict(
            torch.load(crit_path, map_location=device)
        )
        print(f"  Critic: {crit_path}")
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    # Load normalization stats
    norm_stats = None
    norm_path = "data/norm_stats_v2.npz" if Path("data/norm_stats_v2.npz").exists() else "data/norm_stats.npz"
    if Path(norm_path).exists():
        stats = np.load(norm_path)
        norm_stats = {
            'state_mean': torch.tensor(stats['state_mean'], dtype=torch.float32),
            'state_std': torch.tensor(stats['state_std'], dtype=torch.float32),
            'action_mean': torch.tensor(stats['action_mean'], dtype=torch.float32),
            'action_std': torch.tensor(stats['action_std'], dtype=torch.float32),
        }
        print(f"  Norm stats: {norm_path}")

    return jepa, decoder, critic, norm_stats


def build_state_from_screenshot(screenshot_bytes, score, shot_num=0,
                                 total_birds=5):
    """
    Build a REAL state vector from actual screenshot analysis.
    Detects pig/block/bird positions via color detection and maps
    them to the same coordinate system the JEPA was trained on
    (ARENA=800x400, SLINGSHOT=(100,200), GROUND_Y=0).

    This replaces the fake OfflineSimulator state — the JEPA now
    sees where objects ACTUALLY are in the Unity game.
    """
    from PIL import Image
    from io import BytesIO
    from scipy import ndimage

    img = Image.open(BytesIO(screenshot_bytes))
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h, w = arr.shape[:2]

    # Coordinate mapping: screenshot pixels -> training coordinate system
    # Screenshot: (0,0)=top-left, x=right, y=down, size=918x399
    # Training:   (0,0)=bottom-left, x=right, y=up, size=800x400
    # Slingshot:  screenshot=(165, 283) -> training=(100, 200)
    def px_to_game(px_x, px_y):
        """Convert screenshot pixel coords to training game coords."""
        game_x = px_x * 800.0 / w
        game_y = (h - px_y) * 400.0 / h  # flip Y
        return game_x, game_y

    state = np.zeros(OBS_DIM, dtype=np.float32)

    # === BIRDS (5 x 4) ===
    # Detect red bird pixels on left side
    red_mask = (r > 180) & (g < 80) & (b < 80)
    red_mask[:, int(w * 0.4):] = False
    red_ys, red_xs = np.where(red_mask)

    sling_gx, sling_gy = px_to_game(165, 283)  # known slingshot position

    for i in range(5):
        offset = i * 4
        state[offset + 0] = (i % 5) / 4.0  # bird type
        state[offset + 1] = sling_gx / 800.0
        state[offset + 2] = sling_gy / 400.0
        state[offset + 3] = 1.0 if i < shot_num else 0.0

    # === PIGS (detected from screenshot) ===
    pig_mask = (g > 120) & (g > r + 40) & (g > b + 40) & (r < 140) & (b < 100)
    pig_mask[:, :int(w * 0.35)] = False
    pig_mask[int(h * 0.85):, :] = False
    pig_ys, pig_xs = np.where(pig_mask)

    detected_pigs = []
    if len(pig_xs) > 30:
        labeled, n_pigs = ndimage.label(pig_mask)
        for i in range(1, n_pigs + 1):
            py, px = np.where(labeled == i)
            if len(px) > 10:
                cx, cy = float(np.mean(px)), float(np.mean(py))
                gx, gy = px_to_game(cx, cy)
                detected_pigs.append({"x": gx, "y": gy, "size": len(px)})
                if len(detected_pigs) >= 5:
                    break
    elif len(pig_xs) > 5:
        cx, cy = float(np.mean(pig_xs)), float(np.mean(pig_ys))
        gx, gy = px_to_game(cx, cy)
        detected_pigs.append({"x": gx, "y": gy, "size": len(pig_xs)})

    pig_offset = 140  # 20 (birds) + 120 (blocks)
    for i in range(5):
        off = pig_offset + i * 4
        if i < len(detected_pigs):
            pig = detected_pigs[i]
            state[off + 0] = 0.5  # medium size
            state[off + 1] = pig["x"] / 800.0
            state[off + 2] = pig["y"] / 400.0
            state[off + 3] = 1.0  # full health
        # else: zeros (padding)

    # === BLOCKS (detected from screenshot) ===
    detected_blocks = []

    # Wood
    wood = (r > 130) & (r < 220) & (g > 80) & (g < 170) & (b > 30) & (b < 120) & (r > b + 30)
    wood[:, :int(w * 0.35)] = False
    wood[int(h * 0.85):, :] = False

    # Ice
    ice = (b > 150) & (b > r + 20) & (b > g) & (r < 200) & (g > 100)
    ice[:, :int(w * 0.35)] = False
    ice[int(h * 0.85):, :] = False

    # Stone
    stone = (r > 100) & (r < 180) & \
            (np.abs(r.astype(int) - g.astype(int)) < 30) & \
            (np.abs(g.astype(int) - b.astype(int)) < 30) & (r > 80)
    stone[:, :int(w * 0.35)] = False
    stone[int(h * 0.85):, :] = False

    for material_name, material_mask, material_idx in [
        ("wood", wood, 0), ("ice", ice, 1), ("stone", stone, 2)
    ]:
        m_ys, m_xs = np.where(material_mask)
        if len(m_xs) > 50:
            # Cluster into individual blocks using connected components
            labeled, n_blocks = ndimage.label(material_mask)
            for bi in range(1, n_blocks + 1):
                by, bx = np.where(labeled == bi)
                if len(bx) > 20:
                    cx, cy = float(np.mean(bx)), float(np.mean(by))
                    gx, gy = px_to_game(cx, cy)
                    detected_blocks.append({
                        "x": gx, "y": gy,
                        "material": material_idx,
                        "type": 0,
                    })

    block_offset = 20  # after birds
    for i in range(20):
        off = block_offset + i * 6
        if i < len(detected_blocks):
            blk = detected_blocks[i]
            state[off + 0] = blk["type"] / 3.0
            state[off + 1] = blk["material"] / 2.0
            state[off + 2] = blk["x"] / 800.0
            state[off + 3] = blk["y"] / 400.0
            state[off + 4] = 0.0  # rotation
            state[off + 5] = 1.0  # full health

    # === SLINGSHOT (2) ===
    state[160] = sling_gx / 800.0
    state[161] = sling_gy / 400.0

    # === GLOBAL (2) ===
    state[162] = score / 50000.0
    state[163] = max(0, (total_birds - shot_num)) / 5.0

    n_pigs = len(detected_pigs)
    n_blocks = len(detected_blocks)
    print(f"  [VISION] Detected {n_pigs} pigs, {n_blocks} blocks from screenshot")
    if detected_pigs:
        for i, p in enumerate(detected_pigs):
            print(f"    Pig {i+1}: game coords ({p['x']:.0f}, {p['y']:.0f})")

    return state


def play_game(client, jepa, decoder, critic, norm_stats, device,
              level: int = 1, max_shots: int = 5):
    """Play a single level using the trained world model."""

    # Open log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"worldmodel_{timestamp}.txt"
    log = open(log_path, "w", encoding="utf-8")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")

    log_print(f"=== Vortaz Labs -- LeWorldModel Live Play Log ===")
    log_print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"\n{'='*60}")
    log_print(f"  LEVEL {level if level > 0 else 'CURRENT'} -- LeWorldModel is thinking...")
    log_print(f"{'='*60}")

    # Load level (skip if level=0, meaning already loaded)
    if level > 0:
        if not client.load_level(level):
            log_print(f"  Failed to load level {level}")
            log.close()
            return

    time.sleep(0.5)

    total_score = 0
    shots_fired = 0

    for shot_num in range(max_shots):
        game_state = client.get_state()
        if game_state != client.STATE_PLAYING:
            if game_state == client.STATE_WON:
                log_print(f"\n  LEVEL CLEARED! Total score: {client.get_score()}")
            elif game_state == client.STATE_LOST:
                log_print(f"\n  Level failed. Score: {client.get_score()}")
            break

        log_print(f"\n--- Shot {shot_num + 1}/{max_shots} ---")

        # Take a fresh screenshot — this is our REAL game perception
        screenshot = client.get_screenshot()
        if screenshot:
            with open("screenshot_debug.png", "wb") as f:
                f.write(screenshot)

        # Build state from REAL screenshot (not fake OfflineSimulator data)
        score = client.get_score()
        state_raw = build_state_from_screenshot(screenshot, score, shot_num)

        # Log detected objects
        pig_offset = 140
        for i in range(5):
            off = pig_offset + i * 4
            if state_raw[off + 3] > 0:
                log.write(f"  Pig {i+1}: x={state_raw[off+1]*800:.0f}, y={state_raw[off+2]*400:.0f}\n")
        block_offset = 20
        n_blocks = sum(1 for i in range(20) if state_raw[block_offset + i * 6 + 5] > 0)
        log.write(f"  Blocks detected: {n_blocks}\n")

        # Normalize
        if norm_stats:
            state_norm = ((torch.tensor(state_raw).unsqueeze(0) -
                          norm_stats['state_mean']) /
                         norm_stats['state_std'])
        else:
            state_norm = torch.tensor(state_raw).unsqueeze(0)

        # Actor optimizes action through world model + critic
        log_print(f"  World model thinking (optimizing through latent space)...")
        start = time.time()

        result = optimize_action(
            state_norm, jepa, critic,
            decoder=decoder, steps=100, num_starts=6,
            obs_raw=state_raw
        )

        think_time = time.time() - start
        action = result["action_raw"]
        angle = action["angle_deg"]
        power = action["power"]
        tap_time = action["tap_time_ms"] / 1000.0

        log_print(f"  Decision ({think_time:.1f}s):")
        log_print(f"    Angle:    {angle:.1f} deg")
        log_print(f"    Power:    {power:.1%}")
        log_print(f"    Tap time: {tap_time:.2f}s")
        log_print(f"    Energy:   {result['energy']:.4f}")

        # Execute shot in the real game!
        pre_score = client.get_score()
        log_print(f"\n  FIRING!")

        client.do_shot_polar(angle, power, tap_time)
        time.sleep(3.0)  # wait for physics

        post_score = client.get_score()
        score_delta = post_score - pre_score
        total_score = post_score
        shots_fired += 1

        log_print(f"  Result: +{score_delta} points (total: {total_score})")

    log_print(f"\n=== FINAL ===")
    log_print(f"Score: {total_score}")
    log_print(f"Shots fired: {shots_fired}")
    log.close()
    print(f"\n  Log saved to: {log_path}")

    return total_score


def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    VORTAZ LABS — LeWorldModel Live Player         ║
    ║    (LIVE UNITY GAMEPLAY)                         ║
    ║    Watch your world model play Angry Birds!       ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load models
    jepa, decoder, critic, norm_stats = load_models(device)

    # Start WebSocket server and wait for Science Birds to connect
    print("\nStarting WebSocket server on ws://localhost:9000/ ...")
    print("(The Unity game will connect to us as a client)")
    print()

    client = ScienceBirdsClient(host="0.0.0.0", port=9000)

    if not client.connect(timeout=30):
        print("\n" + "="*60)
        print("SCIENCE BIRDS DID NOT CONNECT")
        print("="*60)
        print("""
Make sure the Science Birds game is running in Unity:

  1. Open Unity with the science-birds project
  2. Open scene: Assets/Scenes/GameWorld.unity
  3. Press the PLAY button (triangle) in Unity
  4. The game will auto-connect to this server

If Unity is already in Play mode, try:
  - Stop and re-Press Play in Unity (to reconnect)
  - Check Unity Console for WebSocket errors
        """)

        # Offer offline demo instead
        print("="*60)
        print("Running OFFLINE demo instead (no visual, just physics)")
        print("="*60)
        run_offline_demo(jepa, decoder, critic, norm_stats, device)
        return

    print("Connected! Starting game...\n")

    # Check current state
    time.sleep(1.0)
    state = client.get_state()
    print(f"  Current game state: {state}")

    if state == client.STATE_PLAYING or state == "GameWorld":
        # Already on a level — play it directly
        print("  Game is already on a level — playing it now!")
        score = play_game(
            client, jepa, decoder, critic, norm_stats, device,
            level=0, max_shots=5
        )
    else:
        print(f"  Not on a level yet (state={state}). Navigate to a level in Unity first.")
        print("  Or trying to load level 1...")
        score = play_game(
            client, jepa, decoder, critic, norm_stats, device,
            level=1, max_shots=5
        )

    client.disconnect()
    print("\nDone! Disconnected from Science Birds.")


def run_offline_demo(jepa, decoder, critic, norm_stats, device):
    """Run a visual demo using the offline simulator (no Unity needed)."""
    from science_birds.client import OfflineSimulator
    from science_birds.state_parser import parse_state

    print("\nPlaying 3 demo levels with OfflineSimulator...\n")

    for level_num in range(1, 4):
        sim = OfflineSimulator(seed=100 + level_num)
        level = sim.generate_level("medium")

        print(f"Level {level_num}: {len(level['birds'])} birds, "
              f"{len(level['blocks'])} blocks, {len(level['pigs'])} pigs")

        current_level = level
        total_score = 0

        for bird_i in range(len(level['birds'])):
            pigs_alive = sum(1 for p in current_level['pigs'] if p['health'] > 0)
            if pigs_alive == 0:
                print(f"  ALL PIGS DESTROYED! Score: {total_score}")
                break

            # Build state and optimize
            state_raw = parse_state(current_level)
            if norm_stats:
                state_norm = ((torch.tensor(state_raw).unsqueeze(0) -
                              norm_stats['state_mean']) /
                             norm_stats['state_std'])
            else:
                state_norm = torch.tensor(state_raw).unsqueeze(0)

            result = optimize_action(
                state_norm, jepa, critic,
                decoder=decoder, steps=150, num_starts=6,
                obs_raw=state_raw
            )

            action = result["action_raw"]
            angle = action["angle_deg"]
            power = action["power"]
            tap_s = action["tap_time_ms"] / 1000.0

            # Execute in simulator
            shot = sim.simulate_shot(current_level, bird_i, angle, power, tap_s)
            current_level = shot["level"]
            total_score += shot["score_delta"]

            print(f"  Shot {bird_i+1}: angle={angle:.1f} power={power:.0%} "
                  f"-> +{shot['score_delta']} pts "
                  f"(pigs left: {shot['pigs_alive']})")

            if shot["won"]:
                print(f"  LEVEL CLEARED! Score: {total_score}")
                break

        print()


if __name__ == "__main__":
    main()
