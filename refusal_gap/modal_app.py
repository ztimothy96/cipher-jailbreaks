"""Modal chat-generation class for the refusal-gap experiment.

Setup: see common/modal_infra.py docstring.
"""

import modal

from common.modal_infra import (DEFAULT_MODEL, GPU, HF_CACHE_PATH, app,
                                 hf_cache_volume, image, load_model)


@app.cls(
    image=image,
    gpu=GPU,
    timeout=600,
    scaledown_window=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    retries=
    3,  # transient infra failures (preemption, network) get retried automatically per-request
)
class ChatModel:
    # A modal.parameter makes model_name part of the instance's identity: a
    # ChatModel(model_name="a") and ChatModel(model_name="b") spin up as
    # separate containers rather than silently sharing state. Callers should
    # always pass this explicitly (see measure_refusal_gap.py) and record it
    # in the output alongside every completion.
    model_name: str = modal.parameter(default=DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = load_model(self.model_name)

    @modal.method()
    def generate(self,
                 system_prompt: str,
                 user_turn: str,
                 max_new_tokens: int = 256) -> str:
        import torch

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
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)
