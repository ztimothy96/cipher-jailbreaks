"""Modal activation-extraction class for the probe-generalization experiment.

Captures the residual stream at the prompt's last token position (pre-
generation) for a chosen set of layers, for probe training/eval. Shares the
app/image/HF-cache volume with refusal_gap via common/modal_infra.py.

TODO: implement ActivationExtractor.extract() using forward hooks at the
layer indices under test (see docs/probe-generalization-plan.md once written).
"""

from common.modal_infra import (DEFAULT_MODEL, GPU, HF_CACHE_PATH, app,
                                 hf_cache_volume, image, load_model)
