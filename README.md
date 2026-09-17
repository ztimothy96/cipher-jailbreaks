# cipher-jailbreaks
This project aims to understand how cipher-based jailbreak attacks bypass model refusal mechanisms. It initially studies small models (Qwen2.5-7b-it), hoping to find behaviors that generalize to more powerful models, while addressing two secondary goals of a public personal project:
- save compute budget, and
- avoid jailbreaking large models that could actually cause damage.

![polyrefuse-ablation](results/probe_generalization/ablation_summary__Qwen__Qwen2.5-7B-Instruct__judge__languages__even_layers.png)

There are two main sections of the experiment.
- probe_generalization: reproduces the PolyRefuse paper (a refusal direction that generalizes across natural languages by ablating layers), and extends the same probe/ablation machinery to cipher formats — this is the sole generator of behavioral data (baseline and ablated) for both tracks.
- refusal_gap: the cipher decode-comprehension screen. Checks whether the model can actually decode a cipher back to English at all (scored against ground truth), before trusting any probe or ablation result on that format.

## Structure
The experimental pipeline consists of several standalone Python scripts. Raw model responses and judged scores are written to output .jsonl files, then summary tables and figures are derived from them.
- All behavioral generation (baseline and ablated, languages and ciphers) is written under `results/probe_generalization/` by `probe_generalization/ablate.py` — pass `--layers ""` for baseline-only. `results/refusal_gap/` holds only decode-comprehension-check output.

## Current findings
On Qwen2.5-7b.
- Replication of the PolyRefuse paper. Probes at all layers decode harmfulness of prompts, but only middle layers (esp. layer 16) are causally relevant.
- Partially successful jailbreak on the tiny model using letter-spaced text.
- Ablation on multilingual refusal direction also appears effective in letter-spaced format (need better metrics to quantify this).
- The model seems to frequently echo the prompt instead of refusing or complying.