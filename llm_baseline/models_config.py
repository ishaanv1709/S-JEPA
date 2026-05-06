"""
LLM Models Configuration — Groq API endpoints and model specs.

Uses the Groq Python SDK (pip install groq).
"""

GROQ_MODELS = {
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        "display_name": "GPT-OSS 120B",
        "provider": "OpenAI (via Groq)",
        "context_window": 8192,
        "is_reasoning": True,
    },
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b",
        "display_name": "GPT-OSS 20B",
        "provider": "OpenAI (via Groq)",
        "context_window": 8192,
        "is_reasoning": True,
    },
    "llama-3.1-8b": {
        "model_id": "llama-3.1-8b-instant",
        "display_name": "Llama 3.1 8B",
        "provider": "Meta",
        "context_window": 128000,
        "is_reasoning": False,
    },
    "llama-4-scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "display_name": "Llama 4 Scout 17B",
        "provider": "Meta",
        "context_window": 128000,
        "is_reasoning": False,
    },
}

# Fine-tuned models (local inference, trained on same 50K trajectories)
FINETUNED_MODELS = {
    "qwen3b-finetuned": {
        "display_name": "Qwen2.5-3B (Fine-tuned)",
        "params": "3B",
        "adapter_path": "checkpoints/qwen3b_finetuned",
        "merged_path": "checkpoints/qwen3b_finetuned/merged",
        "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "is_finetuned": True,
        "training_data": "50K trajectories (same as JEPA)",
    },
    "llama1b-finetuned": {
        "display_name": "Llama-3.2-1B (Fine-tuned)",
        "params": "1B",
        "adapter_path": "checkpoints/llama1b_finetuned",
        "merged_path": "checkpoints/llama1b_finetuned/merged",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "is_finetuned": True,
        "training_data": "50K trajectories (same as JEPA)",
    },
}

DEFAULT_BENCHMARK_MODELS = [
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-8b",
    "llama-4-scout",
]

DEFAULT_FINETUNED_MODELS = [
    "qwen3b-finetuned",
    "llama1b-finetuned",
]

