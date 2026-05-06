"""
Vortaz Labs — Fine-Tuned LLM Live Player (Base Module)
========================================================
Shared logic for fine-tuned LLM live play in Unity.
Same structure as play_llm_base.py but uses LOCAL model inference.

The fine-tuned LLM receives:
  1. A screenshot-based text description of the level
  2. Physics context baked into its weights (from training)
  3. Must respond with {angle, power, tap_time} JSON

The shot is executed in Unity via the same API as the world model.
"""

import time
import sys
import os
import re
import json
import numpy as np
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import ScienceBirdsClient
from llm_baseline.finetuned_llm_agent import FinetunedLLMAgent, FINETUNED_MODELS
from play_llm_base import analyze_screenshot, build_live_state_description, parse_llm_shot


def play_with_finetuned_llm(model_key: str, max_shots: int = 5):
    """
    Play the current Science Birds level using a fine-tuned LLM.
    """
    model_info = FINETUNED_MODELS[model_key]
    model_name = model_info["display_name"]

    print(f"""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    VORTAZ LABS — Fine-Tuned LLM Live Player      ║
    ║    (LIVE UNITY GAMEPLAY)                         ║
    ║    Model: {model_name:<40s}║
    ║    (Trained on 50K trajectories — FAIR test)      ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Create local agent
    print(f"  Initializing local model...")
    agent = FinetunedLLMAgent(model_key=model_key)
    print(f"  Model: {model_name} ({model_info['params']})")

    # Connect to Science Birds
    print(f"\n  Starting WebSocket server on ws://localhost:9000/ ...")
    client = ScienceBirdsClient(host="0.0.0.0", port=9000)

    if not client.connect(timeout=30):
        print("  ERROR: Science Birds did not connect!")
        print("  Make sure Unity is in Play mode.")
        return None

    print("  Connected to Science Birds!\n")
    time.sleep(1.0)

    state = client.get_state()
    print(f"  Game state: {state}")

    if state != client.STATE_PLAYING and state != "GameWorld":
        print(f"  Not on a level. Navigate to a level in Unity first.")
        client.disconnect()
        return None

    # Open log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"finetuned_{model_key}_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"=== {model_name} — Fine-Tuned Live Play Log ===\n")
    log_file.write(f"Model: {model_key} ({model_info['params']})\n")
    log_file.write(f"Training: 50K trajectories (same as JEPA)\n")
    log_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    total_score = 0
    results = []

    for shot_num in range(max_shots):
        game_state = client.get_state()
        if game_state != client.STATE_PLAYING:
            if game_state == client.STATE_WON:
                print(f"\n  LEVEL CLEARED! Score: {client.get_score()}")
            elif game_state == client.STATE_LOST:
                print(f"\n  Level failed. Score: {client.get_score()}")
            break

        print(f"\n  Shot {shot_num + 1}/{max_shots}")
        print(f"  {'─' * 40}")

        # Take screenshot and analyze
        screenshot = client.get_screenshot()
        if screenshot:
            with open("screenshot_debug.png", "wb") as f:
                f.write(screenshot)
            screenshot_info = analyze_screenshot(screenshot)
        else:
            screenshot_info = {"width": 918, "height": 399, "pigs": [],
                              "structures": [], "birds": [],
                              "slingshot": {"x": 165, "y": 283}}

        # Build text description
        score = client.get_score()
        state_desc = build_live_state_description(screenshot_info, score, shot_num)
        print(f"  State:\n    " + state_desc.replace("\n", "\n    "))

        # Build prompt and call local model
        # MUST EXACTLY match the training instruction in finetune_llm_data_prep.py
        prompt = (
            f"You are an expert Angry Birds player. Analyze this state and choose the optimal shot.\n\n"
            f"State:\n{state_desc}\n\n"
            f"Respond with ONLY JSON: {{\"angle\": <0-90>, \"power\": <0-100>, \"tap_time\": <0-3>}}"
        )

        print(f"\n  Asking {model_name} (local inference)...")
        log_file.write(f"--- Shot {shot_num + 1} ---\n")
        log_file.write(f"State:\n{state_desc}\n\n")

        start_time = time.time()
        raw_response = agent._generate(prompt, max_new_tokens=128)
        elapsed = time.time() - start_time

        print(f"  Response ({elapsed:.1f}s): {raw_response[:200]}")
        log_file.write(f"Raw response:\n{raw_response}\n\n")

        action = parse_llm_shot(raw_response)
        angle = action["angle"]
        power = action["power"]
        tap_time = action["tap_time"]

        print(f"\n  Decision:")
        print(f"    Angle:    {angle:.1f}°")
        print(f"    Power:    {power:.0%}")
        print(f"    Tap time: {tap_time:.2f}s")
        print(f"    Latency:  {elapsed:.1f}s")

        log_file.write(f"Parsed: angle={angle}, power={power}, tap_time={tap_time}\n\n")

        # Execute shot
        pre_score = client.get_score()
        print(f"\n  FIRING! Watch Unity...")

        client.do_shot_polar(angle, power, tap_time)
        time.sleep(3.0)

        post_score = client.get_score()
        score_delta = post_score - pre_score
        total_score = post_score

        print(f"  Result: +{score_delta} points (total: {total_score})")
        log_file.write(f"Result: +{score_delta} points (total: {total_score})\n\n")

        results.append({
            "shot": shot_num + 1,
            "angle": angle,
            "power": power,
            "tap_time": tap_time,
            "score_delta": score_delta,
            "latency": elapsed,
            "raw_response": raw_response[:500],
        })

    # Summary
    final_score = client.get_score()
    final_state = client.get_state()

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — FINAL RESULTS")
    print(f"{'=' * 50}")
    print(f"  Final score: {final_score}")
    print(f"  Level {'CLEARED' if final_state == client.STATE_WON else 'not cleared'}")
    print(f"  Shots fired: {len(results)}")
    for r in results:
        print(f"    Shot {r['shot']}: {r['angle']:.1f}° {r['power']:.0%} "
              f"-> +{r['score_delta']}pts ({r['latency']:.1f}s)")

    log_file.write(f"\n=== FINAL ===\n")
    log_file.write(f"Score: {final_score}\n")
    log_file.write(f"Cleared: {final_state == client.STATE_WON}\n")
    log_file.write(f"Shots fired: {len(results)}\n")
    log_file.close()

    print(f"\n  Raw outputs saved to: {log_path}")

    client.disconnect()
    return {"model": model_key, "score": final_score,
            "cleared": final_state == client.STATE_WON, "shots": results}
