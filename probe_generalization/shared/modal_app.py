"""Modal classes for the probe-generalization experiment: activation
extraction (Phase A/B). Shares the app/image/HF-cache volume with
refusal_gap via common/modal_infra.py.
"""

import modal
import numpy as np
import torch

from common.modal_infra import (
    DEFAULT_MODEL,
    GPU,
    HF_CACHE_PATH,
    app,
    hf_cache_volume,
    image,
    load_model,
)


@app.cls(
    image=image,
    gpu=GPU,
    timeout=600,
    scaledown_window=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    retries=3,
)
class ActivationExtractor:
    model_name: str = modal.parameter(default=DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = load_model(self.model_name)

    @modal.method()
    def extract(self, system_prompt: str, user_turn: str):
        """One forward pass (no generation): returns the last-token
        residual-stream vector at every layer, as an (num_layers+1,
        hidden_dim) float32 array — index 0 is the embedding output, index i
        is the output of transformer block i. See docs/probe-generalization-plan.md §8."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_turn
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to("cuda")

        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True)

        last_token_vectors = np.stack(
            [h[0, -1, :].float().cpu().numpy() for h in output.hidden_states])
        return last_token_vectors


def _make_ablation_hook(direction):
    """Directional ablation (Arditi et al.): removes the component along
    `direction` (unit vector, torch) from every token position of a
    transformer block's output, each time the block runs during generation."""

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        proj = (hidden @ direction).unsqueeze(-1) * direction
        hidden = hidden - proj
        return (hidden, ) + output[1:] if isinstance(output, tuple) else hidden

    return hook


@app.cls(
    image=image,
    gpu=GPU,
    timeout=600,
    scaledown_window=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    retries=3,
)
class AblationChatModel:
    """Same chat-generation setup as common.chat_model.ChatModel, but
    optionally hooks one transformer block during generation to zero out a
    given direction."""
    model_name: str = modal.parameter(default=DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = load_model(self.model_name)

    @modal.method()
    def generate(self,
                 system_prompt: str,
                 user_turn: str,
                 block_idx: int = None,
                 direction: list = None,
                 max_new_tokens: int = 256) -> str:
        """Ablates direction at the specified block. Runs unmodified baseline if block_idx=None or direction=None."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_turn
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to("cuda")

        handle = None
        if block_idx is not None and direction is not None:
            direction_t = torch.tensor(direction,
                                       dtype=self.model.dtype,
                                       device="cuda")
            handle = self.model.model.layers[block_idx].register_forward_hook(
                _make_ablation_hook(direction_t))

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()

        completion_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)
