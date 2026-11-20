"""Find which Hugging Face models actually work for your token (1 tiny call each).
    python -m tests.hf_check
    python -m tests.hf_check Qwen/Qwen2.5-7B-Instruct other/model
"""
import sys

from huggingface_hub import InferenceClient

from app.config import get_settings
from app.llm import describe_llm_error

DEFAULTS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]


def main(models):
    s = get_settings()
    print(f"token set: {bool(s.hf_token)} | configured provider: {s.hf_provider}\n")
    for provider in ("auto", "hf-inference"):
        for m in models:
            try:
                client = InferenceClient(provider=provider, api_key=s.hf_token, timeout=60)
                out = client.chat_completion(
                    model=m, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5)
                print(f"OK    {provider:13} {m}: {out.choices[0].message.content!r}")
            except Exception as e:
                print(f"FAIL  {provider:13} {m}: {describe_llm_error(e)[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULTS)
