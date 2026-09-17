"""
Shared Modal infrastructure for all experiments in this repo: one App, one
base image, one HF-weights cache volume. Experiment-specific `@app.cls`
definitions (e.g. src/shared/modal_app.py) import from here rather than
each declaring their own app/image/volume, so containers across
experiments share the cached model weights and don't fragment the Modal
app namespace.

Auth: run `pip install modal && modal setup` once (interactive browser login)
before using anything here.
"""

import modal

APP_NAME = "cipher-jailbreaks"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU_BY_MODEL = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "A10G",
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "google/gemma-2-9b-it": "A10G",
    "mistralai/Mistral-7B-Instruct-v0.3": "A10G",
    "Qwen/Qwen2.5-14B-Instruct": "A100-40GB",
    "Qwen/Qwen2.5-32B-Instruct": "A100-80GB",
}
# Fallback for an unlisted model
DEFAULT_GPU = "A100-40GB"


def gpu_for(model_name: str) -> str:
    """GPU tier for a given model. See GPU_BY_MODEL above."""
    gpu = GPU_BY_MODEL.get(model_name)
    if gpu is None:
        print(f"Warning: no GPU tier configured for '{model_name}' in "
              f"common/modal_infra.py's GPU_BY_MODEL — defaulting to "
              f"{DEFAULT_GPU}. Add an entry once you know what fits.")
        return DEFAULT_GPU
    return gpu


app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.44",
    "accelerate",
    "sentencepiece",
).add_local_python_source("common", "src")

# Persists the Hugging Face cache across runs/cold-starts, and across
# experiments, so the ~15GB model download only happens once.
hf_cache_volume = modal.Volume.from_name("cipher-jailbreaks-hf-cache",
                                         create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
CANDIDATE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
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
