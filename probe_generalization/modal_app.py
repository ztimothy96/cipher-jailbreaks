"""Modal classes for the probe-generalization experiment: NLLB translation
(Phase A language rendering) and activation extraction (Phase A/B). Shares
the app/image/HF-cache volume with refusal_gap via common/modal_infra.py.
"""

import modal
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common.modal_infra import (
    DEFAULT_MODEL,
    GPU,
    HF_CACHE_PATH,
    app,
    hf_cache_volume,
    image,
    load_model,
)

NLLB_MODEL = "facebook/nllb-200-3.3B"


@app.cls(
    image=image,
    gpu=GPU,
    timeout=600,
    scaledown_window=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    retries=3,
)
class Translator:
    """Dedicated translation model (not a chat/instruction-tuned LLM)."""

    model_name: str = modal.parameter(default=NLLB_MODEL)

    @modal.enter()
    def load(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    @modal.method()
    def translate(self,
                  text: str,
                  src_lang: str,
                  tgt_lang: str,
                  max_new_tokens: int = 256) -> str:
        """src_lang/tgt_lang are FLORES-200 codes, e.g. 'eng_Latn',
        'zho_Hans' — see probe_generalization/formats.py FLORES_CODES."""

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
