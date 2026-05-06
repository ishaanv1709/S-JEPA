"""
Vortaz Labs — Fine-Tuned Qwen2.5-3B plays Angry Birds LIVE
=============================================================
Run: python play_finetuned_qwen3b.py

This model was fine-tuned on the SAME 50K trajectories as the
JEPA world model — a FAIR comparison, unlike zero-shot LLMs.

Make sure Unity Science Birds is in Play mode first.
"""
from play_finetuned_base import play_with_finetuned_llm

if __name__ == "__main__":
    play_with_finetuned_llm("qwen3b-finetuned", max_shots=5)
