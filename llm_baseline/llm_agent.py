"""
LLM Agent — Unified interface for Groq-hosted LLMs playing Angry Birds.

Uses the Groq Python SDK (pip install groq).
Each LLM receives the same game state as the JEPA world model (converted
to text) and must choose actions or predict outcomes via text reasoning.

Rotates across multiple GROQ_API_KEY_1..N to avoid rate limits.
"""

import os
import time
import json
from typing import Optional
from pathlib import Path

# Load .env file if it exists
env_path = Path(__file__).resolve().parent.parent / ".env"
_env_vars = {}
if env_path.exists():
    for line in env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            val = val.strip().strip('"').strip("'")
            _env_vars[key.strip()] = val
            os.environ.setdefault(key.strip(), val)

# Collect all API keys for rotation
_API_KEYS = []
for i in range(1, 20):
    k = _env_vars.get(f"GROQ_API_KEY_{i}") or os.environ.get(f"GROQ_API_KEY_{i}")
    if k:
        _API_KEYS.append(k)
if not _API_KEYS:
    # Fallback to single key
    k = _env_vars.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if k:
        _API_KEYS.append(k)

_key_index = 0

def _next_key():
    """Round-robin through API keys."""
    global _key_index
    key = _API_KEYS[_key_index % len(_API_KEYS)]
    _key_index += 1
    return key

from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_baseline.models_config import GROQ_MODELS
from llm_baseline.prompt_builder import build_action_prompt, build_prediction_prompt
from llm_baseline.response_parser import parse_action_response, parse_prediction_response


class GroqLLMAgent:
    """
    LLM agent that plays Angry Birds via Groq SDK.

    Fair comparison with JEPA:
    - Receives same state information (converted to text)
    - Must produce same output format (angle, power, tap_time)
    - Latency is measured (JEPA inference vs LLM generation)
    """

    def __init__(self, model_key: str = "gpt-oss-120b",
                 temperature: float = 0.3,
                 max_retries: int = 4):
        self.model_config = GROQ_MODELS[model_key]
        self.model_id = self.model_config["model_id"]
        self.model_name = self.model_config["display_name"]
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize Groq client with key rotation
        self.client = Groq(api_key=_next_key())
        self.is_reasoning = self.model_config.get("is_reasoning", "gpt-oss" in self.model_id)

    def _rotate_key(self):
        """Switch to next API key (for rate limit avoidance)."""
        self.client = Groq(api_key=_next_key())

    def _call_api(self, messages: list) -> dict:
        """Call Groq API via SDK and return response with timing."""
        start_time = time.time()

        try:
            if self.is_reasoning:
                # Reasoning model: streaming + reasoning_effort
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=4096,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=True,
                    stop=None,
                )
                content_parts = []
                for chunk in completion:
                    content_parts.append(chunk.choices[0].delta.content or "")
                content = "".join(content_parts)
                latency_ms = (time.time() - start_time) * 1000

                return {
                    "content": content,
                    "latency_ms": latency_ms,
                    "tokens_used": 0,
                }
            else:
                # Standard model: non-streaming
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=1024,
                    top_p=1,
                )
                latency_ms = (time.time() - start_time) * 1000
                content = completion.choices[0].message.content or ""
                usage = completion.usage

                return {
                    "content": content,
                    "latency_ms": latency_ms,
                    "tokens_used": usage.total_tokens if usage else 0,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }

        except Exception as e:
            # Rotate key on rate limit before returning error
            if "429" in str(e) or "rate" in str(e).lower():
                self._rotate_key()
            latency_ms = (time.time() - start_time) * 1000
            return {
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "latency_ms": latency_ms,
                "tokens_used": 0,
            }

    def choose_action(self, ground_truth: dict) -> dict:
        """
        Ask LLM to choose the best shot for the current game state.

        Returns:
            dict with action, latency_ms, tokens_used, raw_response
        """
        messages = build_action_prompt(ground_truth)

        for attempt in range(self.max_retries):
            api_result = self._call_api(messages)

            if "error" in api_result:
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * 2 + 2
                    print(f"    [{self.model_name}] API error, retrying in {sleep_time}s: "
                          f"{api_result['error'][:80]}")
                    time.sleep(sleep_time)
                    continue
                return {
                    "action": {"angle": 45.0, "power": 0.7, "tap_time": 0.0},
                    "latency_ms": api_result["latency_ms"],
                    "tokens_used": 0,
                    "error": api_result["error"],
                    "raw_response": "",
                    "parse_success": False,
                }

            parsed = parse_action_response(api_result["content"])
            if parsed is not None:
                return {
                    "action": parsed,
                    "latency_ms": api_result["latency_ms"],
                    "tokens_used": api_result.get("tokens_used", 0),
                    "raw_response": api_result["content"],
                    "parse_success": True,
                }

            # Parse failed, retry
            if attempt < self.max_retries - 1:
                continue

        # All retries failed — use default action
        return {
            "action": {"angle": 45.0, "power": 0.7, "tap_time": 0.0},
            "latency_ms": api_result.get("latency_ms", 0),
            "tokens_used": api_result.get("tokens_used", 0),
            "raw_response": api_result.get("content", ""),
            "parse_success": False,
        }

    def predict_outcome(self, ground_truth: dict,
                        angle: float, power: float,
                        tap_time: float) -> dict:
        """
        Ask LLM to PREDICT the outcome of a specific shot.
        Tests world modeling ability directly.
        """
        messages = build_prediction_prompt(
            ground_truth, angle, power * 100, tap_time
        )

        for attempt in range(self.max_retries):
            api_result = self._call_api(messages)

            if "error" in api_result:
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * 2 + 2
                    time.sleep(sleep_time)
                    continue
                return {
                    "prediction": None,
                    "latency_ms": api_result["latency_ms"],
                    "tokens_used": 0,
                    "error": api_result["error"],
                }

            parsed = parse_prediction_response(api_result["content"])
            if parsed is not None:
                return {
                    "prediction": parsed,
                    "latency_ms": api_result["latency_ms"],
                    "tokens_used": api_result.get("tokens_used", 0),
                    "raw_response": api_result["content"],
                }

            if attempt < self.max_retries - 1:
                continue

        return {
            "prediction": None,
            "latency_ms": api_result.get("latency_ms", 0),
            "tokens_used": api_result.get("tokens_used", 0),
            "raw_response": api_result.get("content", ""),
        }


if __name__ == "__main__":
    from science_birds.client import OfflineSimulator

    sim = OfflineSimulator(seed=42)
    level = sim.generate_level("medium")

    for model_key in ["gpt-oss-120b"]:
        print(f"\n=== {GROQ_MODELS[model_key]['display_name']} ===")
        agent = GroqLLMAgent(model_key=model_key)

        result = agent.choose_action(level)
        print(f"Action: {result['action']}")
        print(f"Latency: {result['latency_ms']:.0f}ms")
        print(f"Tokens: {result['tokens_used']}")
        print(f"Parse OK: {result['parse_success']}")
        if result.get("raw_response"):
            print(f"Raw: {result['raw_response'][:200]}")
