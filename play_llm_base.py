"""
Vortaz Labs — LLM Live Player (Base Module)
=============================================
Shared logic for all LLM players. Each model-specific script
imports from here and just sets the model key.

The LLM receives:
  1. A screenshot-based text description of the level
  2. Physics rules and bird ability info
  3. Must respond with {angle, power, tap_time} JSON

The shot is executed in Unity via the same shootbird command
as the world model, for a fair visual comparison.
"""

import time
import sys
import os
import re
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from science_birds.client import ScienceBirdsClient
from llm_baseline.llm_agent import GroqLLMAgent
from llm_baseline.models_config import GROQ_MODELS


def analyze_screenshot(screenshot_bytes: bytes) -> dict:
    """
    Analyze the game screenshot to extract object positions.
    Returns a dict describing what's visible in the level.
    """
    img = Image.open(BytesIO(screenshot_bytes))
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h, w = arr.shape[:2]

    info = {"width": w, "height": h, "pigs": [], "structures": [], "birds": []}

    # === Find PIGS (bright green, right half, above ground) ===
    pig_mask = (g > 120) & (g > r + 40) & (g > b + 40) & (r < 140) & (b < 100)
    pig_mask[:, :int(w * 0.35)] = False
    pig_mask[int(h * 0.85):, :] = False
    pig_ys, pig_xs = np.where(pig_mask)

    if len(pig_xs) > 30:
        # Cluster pigs by proximity
        from scipy import ndimage
        labeled, n_pigs = ndimage.label(pig_mask)
        for i in range(1, n_pigs + 1):
            py, px = np.where(labeled == i)
            if len(px) > 10:
                info["pigs"].append({
                    "x": int(np.mean(px)),
                    "y": int(np.mean(py)),
                    "size": len(px),
                })
    elif len(pig_xs) > 5:
        info["pigs"].append({
            "x": int(np.mean(pig_xs)),
            "y": int(np.mean(pig_ys)),
            "size": len(pig_xs),
        })

    # === Find structures (wood/ice/stone on right half) ===
    wood = (r > 130) & (r < 220) & (g > 80) & (g < 170) & (b > 30) & (b < 120) & (r > b + 30)
    wood[:, :int(w * 0.35)] = False
    wood[int(h * 0.85):, :] = False
    wood_ys, wood_xs = np.where(wood)

    if len(wood_xs) > 30:
        info["structures"].append({
            "material": "wood",
            "x_center": int(np.mean(wood_xs)),
            "y_center": int(np.mean(wood_ys)),
            "x_min": int(np.min(wood_xs)),
            "x_max": int(np.max(wood_xs)),
            "y_min": int(np.min(wood_ys)),
            "y_max": int(np.max(wood_ys)),
            "pixels": len(wood_xs),
        })

    ice = (b > 150) & (b > r + 20) & (b > g) & (r < 200) & (g > 100)
    ice[:, :int(w * 0.35)] = False
    ice[int(h * 0.85):, :] = False
    ice_ys, ice_xs = np.where(ice)

    if len(ice_xs) > 30:
        info["structures"].append({
            "material": "ice",
            "x_center": int(np.mean(ice_xs)),
            "y_center": int(np.mean(ice_ys)),
            "pixels": len(ice_xs),
        })

    # === Find birds (red, left side) ===
    red_mask = (r > 180) & (g < 80) & (b < 80)
    red_mask[:, int(w * 0.4):] = False
    red_ys, red_xs = np.where(red_mask)
    if len(red_xs) > 5:
        info["birds"].append({
            "type": "red",
            "x": int(np.mean(red_xs)),
            "y": int(np.mean(red_ys)),
        })

    # Bird on slingshot position (known)
    info["slingshot"] = {"x": int(165 * w / 918), "y": int(283 * h / 399)}

    return info


def build_live_state_description(screenshot_info: dict, score: int,
                                  shot_num: int, total_birds: int = 5) -> str:
    """Build a text description from screenshot analysis for the LLM prompt."""
    lines = []
    w, h = screenshot_info["width"], screenshot_info["height"]

    lines.append(f"Screen: {w}x{h} pixels")
    lines.append(f"Score: {score}")
    lines.append(f"Shot: {shot_num + 1} of {total_birds}")
    lines.append(f"Birds remaining: {total_birds - shot_num}")

    sling = screenshot_info["slingshot"]
    lines.append(f"Slingshot position: ({sling['x']}, {sling['y']})")

    pigs = screenshot_info.get("pigs", [])
    if pigs:
        lines.append(f"\nPigs found: {len(pigs)}")
        for i, pig in enumerate(pigs):
            # Convert pixel Y to height (0=ground, higher=taller)
            height_pct = 100 * (h - pig["y"]) / h
            dist_from_sling = pig["x"] - sling["x"]
            lines.append(f"  Pig {i+1}: at pixel ({pig['x']}, {pig['y']}) "
                        f"— {dist_from_sling}px right of slingshot, "
                        f"height={height_pct:.0f}% from bottom")
    else:
        lines.append("\nPigs: not clearly visible (may be behind structures)")

    structs = screenshot_info.get("structures", [])
    if structs:
        lines.append(f"\nStructures:")
        for s in structs:
            dist = s["x_center"] - sling["x"] if "x_center" in s else 0
            height_pct = 100 * (h - s.get("y_center", h//2)) / h
            lines.append(f"  {s['material'].capitalize()} blocks: "
                        f"center at ({s.get('x_center', '?')}, {s.get('y_center', '?')}) "
                        f"— {dist}px right, height={height_pct:.0f}%")

    return "\n".join(lines)


def build_llm_live_prompt(state_desc: str) -> list:
    """Build the chat prompt for the LLM to play live."""
    system = """You are an expert Angry Birds player. You must analyze the game state and choose the optimal shot.

Physics rules:
- The slingshot is on the LEFT side of the screen
- Pigs and structures are on the RIGHT side
- You pull the bird BACK (left) and UP to launch it RIGHT toward the targets
- angle=0° shoots perfectly horizontal (flat), angle=45° is a balanced arc, angle=90° shoots straight up
- Higher angles give more height but less horizontal distance
- Power 100% = maximum distance, 50% = half distance
- Gravity pulls the bird down during flight, so aim HIGHER than the target
- If the pig is HIGH UP, use a HIGH angle (40-60°)
- If the pig is far right and at ground level, use a LOW angle (15-30°) with high power

IMPORTANT: You MUST respond with ONLY a JSON object, nothing else:
{"angle": <float 0-90>, "power": <float 0-100>, "tap_time": <float 0-3>}"""

    user = f"""Current game state from screenshot analysis:
{state_desc}

Based on the pig positions relative to the slingshot, what angle and power should I use?
Remember: if pigs are HIGH UP, use a STEEP angle (40-70°). If pigs are far right on the ground, use a flat angle (15-30°).

Respond with ONLY the JSON: {{"angle": <0-90>, "power": <0-100>, "tap_time": <0-3>}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_llm_shot(raw_response: str) -> dict:
    """
    Robustly extract angle/power/tap_time from LLM response.
    Handles JSON, markdown blocks, extra text, etc.
    """
    # Try clean JSON
    try:
        data = json.loads(raw_response.strip())
        return _validate(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try markdown code block
    m = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', raw_response, re.DOTALL)
    if m:
        try:
            return _validate(json.loads(m.group(1)))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding JSON object with angle key
    m = re.search(r'\{[^{}]*"angle"[^{}]*\}', raw_response, re.DOTALL)
    if m:
        try:
            return _validate(json.loads(m.group()))
        except (json.JSONDecodeError, TypeError):
            pass

    # Regex fallback
    angle_m = re.search(r'angle["\s:]*(\d+\.?\d*)', raw_response)
    power_m = re.search(r'power["\s:]*(\d+\.?\d*)', raw_response)
    tap_m = re.search(r'tap[_\s]*time["\s:]*(\d+\.?\d*)', raw_response)

    if angle_m and power_m:
        return _validate({
            "angle": float(angle_m.group(1)),
            "power": float(power_m.group(1)),
            "tap_time": float(tap_m.group(1)) if tap_m else 0.0,
        })

    # Total fallback
    print(f"  WARNING: Could not parse LLM response, using default 45°/80%")
    return {"angle": 45.0, "power": 0.80, "tap_time": 0.0}


def _validate(data: dict) -> dict:
    angle = float(max(0, min(90, data.get("angle", 45))))
    power_raw = float(max(0, min(100, data.get("power", 80))))
    # Power comes as 0-100 from LLM, normalize to 0-1
    power = power_raw / 100.0 if power_raw > 1.0 else power_raw
    tap = float(max(0, min(3, data.get("tap_time", 0))))
    return {"angle": angle, "power": power, "tap_time": tap}


def play_with_llm(model_key: str, max_shots: int = 5):
    """
    Play the current Science Birds level using an LLM via Groq API.
    """
    model_info = GROQ_MODELS[model_key]
    model_name = model_info["display_name"]

    print(f"""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    VORTAZ LABS — LLM Live Player                 ║
    ║    (LIVE UNITY GAMEPLAY)                         ║
    ║    Model: {model_name:<40s}║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Create LLM agent
    agent = GroqLLMAgent(model_key=model_key)
    print(f"  Model: {model_name} ({model_info['model_id']})")
    print(f"  Reasoning: {'Yes' if model_info.get('is_reasoning') else 'No'}")

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

    # Open log file for raw outputs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"llm_{model_key}_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"=== {model_name} — Live Play Log ===\n")
    log_file.write(f"Model: {model_info['model_id']}\n")
    log_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    total_score = 0
    results = []

    for shot_num in range(max_shots):
        game_state = client.get_state()
        if game_state != client.STATE_PLAYING:
            if game_state == client.STATE_WON:
                print(f"\n  🎉 LEVEL CLEARED! Score: {client.get_score()}")
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

        # Build text description from screenshot
        score = client.get_score()
        state_desc = build_live_state_description(screenshot_info, score, shot_num)
        print(f"  State:\n    " + state_desc.replace("\n", "\n    "))

        # Build prompt and call LLM
        messages = build_llm_live_prompt(state_desc)

        print(f"\n  Asking {model_name}...")
        log_file.write(f"--- Shot {shot_num + 1} ---\n")
        log_file.write(f"State:\n{state_desc}\n\n")
        log_file.write(f"Prompt:\n{messages[-1]['content']}\n\n")

        start_time = time.time()
        api_result = agent._call_api(messages)
        elapsed = time.time() - start_time

        raw_response = api_result.get("content", api_result.get("error", ""))
        print(f"  Response ({elapsed:.1f}s): {raw_response[:200]}")

        log_file.write(f"Raw response:\n{raw_response}\n\n")

        if "error" in api_result:
            print(f"  ERROR: {api_result['error'][:100]}")
            log_file.write(f"ERROR: {api_result['error']}\n\n")
            # Use default
            action = {"angle": 45.0, "power": 0.80, "tap_time": 0.0}
        else:
            action = parse_llm_shot(raw_response)

        angle = action["angle"]
        power = action["power"]
        tap_time = action["tap_time"]

        print(f"\n  Decision:")
        print(f"    Angle:    {angle:.1f}°")
        print(f"    Power:    {power:.0%}")
        print(f"    Tap time: {tap_time:.2f}s")
        print(f"    Latency:  {elapsed:.1f}s")

        log_file.write(f"Parsed: angle={angle}, power={power}, tap_time={tap_time}\n")
        log_file.write(f"Decision: {angle:.1f} deg, {power:.0%} power, {tap_time:.2f}s tap, {elapsed:.1f}s latency\n\n")

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
    for r in results:
        log_file.write(f"  Shot {r['shot']}: {r['angle']:.1f} deg, {r['power']:.0%} power "
                       f"-> +{r['score_delta']}pts ({r['latency']:.1f}s)\n")
    log_file.close()

    print(f"\n  Raw outputs saved to: {log_path}")

    client.disconnect()
    return {"model": model_key, "score": final_score,
            "cleared": final_state == client.STATE_WON, "shots": results}
