"""
Vortaz Labs — Fine-Tuned Llama-3.2-1B plays Angry Birds LIVE
===============================================================
Run: python play_finetuned_llama1b.py

This model was fine-tuned on the SAME 50K trajectories as the
JEPA world model — a FAIR comparison, unlike zero-shot LLMs.

Make sure Unity Science Birds is in Play mode first.
"""
from play_finetuned_base import play_with_finetuned_llm

if __name__ == "__main__":
    play_with_finetuned_llm("llama1b-finetuned", max_shots=5)
