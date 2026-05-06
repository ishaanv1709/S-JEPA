"""
Vortaz Labs — Fine-Tuned LLM Agent (Local Inference)
======================================================
Runs the Unsloth fine-tuned models locally for both offline
evaluation and Unity live play. No API calls needed.

Supports:
  - Qwen2.5-3B (fine-tuned on Science Birds trajectories)
  - Llama-3.2-1B (fine-tuned on Science Birds trajectories)

Same interface as GroqLLMAgent: choose_action() and predict_outcome()
"""

import os
import sys
import time
import json
import re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FINETUNED_MODELS = {
    "qwen3b-finetuned": {
        "display_name": "Qwen2.5-3B (Fine-tuned)",
        "adapter_path": "checkpoints/qwen3b_finetuned",
        "merged_path": "checkpoints/qwen3b_finetuned/merged",
        "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "params": "3B",
        "is_finetuned": True,
    },
    "llama1b-finetuned": {
        "display_name": "Llama-3.2-1B (Fine-tuned)",
        "adapter_path": "checkpoints/llama1b_finetuned",
        "merged_path": "checkpoints/llama1b_finetuned/merged",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "params": "1B",
        "is_finetuned": True,
    },
}


class FinetunedLLMAgent:
    """
    Local inference agent for fine-tuned LLMs.

    Fair comparison with JEPA:
    - Trained on SAME 50K trajectories
    - Receives same state information (as text)
    - Must produce same output format
    - Latency measured for comparison
    """

    def __init__(self, model_key: str = "qwen3b-finetuned", device: str = None):
        self.model_config = FINETUNED_MODELS[model_key]
        self.model_key = model_key
        self.model_name = self.model_config["display_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Loading {self.model_name}...")
        t0 = time.time()

        base_dir = Path(__file__).resolve().parent.parent
        merged_path = base_dir / self.model_config["merged_path"]
        adapter_path = base_dir / self.model_config["adapter_path"]

        if merged_path.exists():
            print(f"  Using merged model: {merged_path}")
            self._load_merged(str(merged_path))
        elif adapter_path.exists():
            print(f"  Using adapter: {adapter_path}")
            self._load_adapter(str(adapter_path))
        else:
            raise FileNotFoundError(
                f"No model found at {merged_path} or {adapter_path}. "
                f"Run the fine-tuning script first."
            )

        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s")

    def _load_merged(self, path: str):
        """Load the merged model (no adapter needed)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def _load_adapter(self, path: str):
        """Load base model + LoRA adapter."""
        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            base_model_name = self.model_config["base_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, path)
            self.model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from the model."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def _parse_json(self, raw_response: str) -> dict:
        """Extract JSON from model response."""
        # Try clean JSON
        try:
            return json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON block
        m = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def choose_action(self, ground_truth: dict) -> dict:
        """Choose action for current game state (for live play)."""
        from science_birds.state_parser import state_to_description
        state_desc = state_to_description(ground_truth)

        prompt = (
            f"You are an expert Angry Birds player. Analyze this state and choose the optimal shot.\n\n"
            f"State:\n{state_desc}\n\n"
            f"Respond with ONLY JSON: {{\"angle\": <0-90>, \"power\": <0-100>, \"tap_time\": <0-3>}}"
        )

        start = time.time()
        raw = self._generate(prompt, max_new_tokens=128)
        latency = (time.time() - start) * 1000

        parsed = self._parse_json(raw)
        if parsed and "angle" in parsed:
            action = {
                "angle": float(max(0, min(90, parsed.get("angle", 45)))),
                "power": float(max(0, min(100, parsed.get("power", 80)))) / 100.0,
                "tap_time": float(max(0, min(3, parsed.get("tap_time", 0)))),
            }
            return {
                "action": action,
                "latency_ms": latency,
                "raw_response": raw,
                "parse_success": True,
            }

        return {
            "action": {"angle": 45.0, "power": 0.7, "tap_time": 0.0},
            "latency_ms": latency,
            "raw_response": raw,
            "parse_success": False,
        }

    def predict_outcome(self, ground_truth: dict, angle: float,
                        power: float, tap_time: float) -> dict:
        """Predict outcome of action on state (for evaluation)."""
        from science_birds.state_parser import state_to_description
        state_text = state_to_description(ground_truth)
        prompt = (
            f"Given the current game state and an action taken by a player, "
            f"predict what the game state will look like after the action is executed.\n\n"
            f"Current State:\n{state_text}\n\n"
            f"Action: angle={angle:.3f}, power={power:.3f}, tap_time={tap_time:.3f}\n\n"
            f"Predict the resulting state as JSON with: pigs_alive, blocks_remaining, "
            f"score_delta, new_score, birds_left, and individual pig health values."
        )

        start = time.time()
        raw = self._generate(prompt, max_new_tokens=256)
        latency = (time.time() - start) * 1000

        parsed = self._parse_json(raw)
        if parsed and "score_delta" in parsed:
            parsed["predicted_score_delta"] = parsed["score_delta"]

        return {
            "prediction": parsed,
            "latency_ms": latency,
            "raw_response": raw,
        }


if __name__ == "__main__":
    print("\n=== Testing Fine-Tuned LLM Agent ===\n")

    for model_key in FINETUNED_MODELS:
        try:
            agent = FinetunedLLMAgent(model_key=model_key)
            # Note: Test is disabled for the generic test block because it requires a true level dictionary object
            print("Successfully instantiated the agent!")
            print(f"\n  {agent.model_name}:")
            print(f"    Latency:  {result['latency_ms']:.0f}ms")
            print(f"    Response: {result['raw_response'][:200]}")
        except Exception as e:
            print(f"\n  {model_key}: {e}")
