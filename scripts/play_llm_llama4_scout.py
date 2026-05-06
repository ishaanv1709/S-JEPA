"""
Vortaz Labs — Llama 4 Scout 17B plays Angry Birds LIVE
========================================================
Run: python play_llm_llama4_scout.py

Make sure Unity Science Birds is in Play mode first.
"""
from play_llm_base import play_with_llm

if __name__ == "__main__":
    play_with_llm("llama-4-scout", max_shots=5)
