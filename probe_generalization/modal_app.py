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
