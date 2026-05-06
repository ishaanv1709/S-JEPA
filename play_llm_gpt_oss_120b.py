"""
Vortaz Labs — GPT-OSS 120B plays Angry Birds LIVE
===================================================
Run: python play_llm_gpt_oss_120b.py

Make sure Unity Science Birds is in Play mode first.
"""
from play_llm_base import play_with_llm

if __name__ == "__main__":
    play_with_llm("gpt-oss-120b", max_shots=5)
