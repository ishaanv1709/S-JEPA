"""
Vortaz Labs — Llama 3.1 8B plays Angry Birds LIVE
===================================================
Run: python play_llm_llama31_8b.py

Make sure Unity Science Birds is in Play mode first.
"""
from play_llm_base import play_with_llm

if __name__ == "__main__":
    play_with_llm("llama-3.1-8b", max_shots=5)
