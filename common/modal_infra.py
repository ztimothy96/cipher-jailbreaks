"""
Shared Modal infrastructure for all experiments in this repo: one App, one
base image, one HF-weights cache volume. Experiment-specific `@app.cls`
definitions (e.g. refusal_gap/modal_app.py, probe_generalization/modal_app.py)
import from here rather than each declaring their own app/image/volume, so
containers across experiments share the cached model weights and don't
fragment the Modal app namespace.

Auth: run `pip install modal && modal setup` once (interactive browser login)
before using anything here.
"""

import modal

APP_NAME = "cipher-jailbreaks"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# A10G: 24GB was fine for 7B but OOMs on 14B.
# A100-40GB: bump this again (and the GPU) if we move to a >~20B model.
GPU = "A100-40GB"

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.44",
    "accelerate",
    "sentencepiece",
).add_local_python_source("common", "refusal_gap", "probe_generalization")

# Persists the Hugging Face cache across runs/cold-starts, and across
# experiments, so the ~15GB model download only happens once.
hf_cache_volume = modal.Volume.from_name("cipher-jailbreaks-hf-cache",
                                         create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
CANDIDATE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def model_slug(model_name: str) -> str:
    """A model name as a filesystem/filename-safe slug, e.g. for result
    file names: 'Qwen/Qwen2.5-7B-Instruct' -> 'Qwen__Qwen2.5-7B-Instruct'."""
    return model_name.replace("/", "__")


def load_model(model_name: str):
    """Load a causal LM + tokenizer onto CUDA. Shared by any Modal class
    that needs the base model (generation, activation extraction, ...)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    return model, tokenizer
