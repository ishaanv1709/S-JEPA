"""
Response Parser — Extracts structured actions from LLM text output.

LLMs often return JSON with extra text, markdown formatting, or
slightly invalid syntax. This parser handles all edge cases robustly.
"""

import json
import re
from typing import Optional


def parse_action_response(text: str) -> Optional[dict]:
    """
    Extract action (angle, power, tap_time) from LLM response text.

    Handles:
    - Clean JSON: {"angle": 45, "power": 80, "tap_time": 1.0}
    - Markdown-wrapped: ```json {...} ```
    - Extra text around JSON
    - Missing fields (fills defaults)
    """
    # Try direct JSON parse first
    try:
        data = json.loads(text.strip())
        return _validate_action(data)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    code_block = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', text, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1))
            return _validate_action(data)
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    json_match = re.search(r'\{[^{}]*"angle"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _validate_action(data)
        except json.JSONDecodeError:
            pass

    # Last resort: regex for numbers
    angle_match = re.search(r'angle["\s:]*(\d+\.?\d*)', text)
    power_match = re.search(r'power["\s:]*(\d+\.?\d*)', text)
    tap_match = re.search(r'tap[_\s]*time["\s:]*(\d+\.?\d*)', text)

    if angle_match and power_match:
        return _validate_action({
            "angle": float(angle_match.group(1)),
            "power": float(power_match.group(1)),
            "tap_time": float(tap_match.group(1)) if tap_match else 0.0,
        })

    return None


def parse_prediction_response(text: str) -> Optional[dict]:
    """Extract prediction from LLM response."""
    try:
        data = json.loads(text.strip())
        return _validate_prediction(data)
    except json.JSONDecodeError:
        pass

    code_block = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', text, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1))
            return _validate_prediction(data)
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*"predicted_score_delta"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _validate_prediction(data)
        except json.JSONDecodeError:
            pass

    return None


def _validate_action(data: dict) -> dict:
    """Validate and normalize action fields."""
    return {
        "angle": float(max(0, min(90, data.get("angle", 45)))),
        "power": float(max(0, min(100, data.get("power", 50)))) / 100.0,
        "tap_time": float(max(0, min(3, data.get("tap_time", 0)))),
    }


def _validate_prediction(data: dict) -> dict:
    """Validate prediction fields."""
    return {
        "predicted_score_delta": int(data.get("predicted_score_delta", 0)),
        "pigs_killed": int(data.get("pigs_killed", 0)),
        "blocks_destroyed": int(data.get("blocks_destroyed", 0)),
        "projectile_landing_x": float(data.get("projectile_landing_x", 400)),
        "reasoning": str(data.get("reasoning", "No reasoning provided")),
    }


if __name__ == "__main__":
    # Test various LLM output formats
    tests = [
        '{"angle": 45, "power": 80, "tap_time": 1.0}',
        '```json\n{"angle": 35.5, "power": 90, "tap_time": 0}\n```',
        'Based on my analysis, the best shot is {"angle": 50, "power": 75, "tap_time": 1.5} because...',
        'angle: 40, power: 85, tap_time: 0.5',
        'totally broken response with no json',
    ]

    for text in tests:
        result = parse_action_response(text)
        print(f"Input:  {text[:60]}...")
        print(f"Parsed: {result}\n")
