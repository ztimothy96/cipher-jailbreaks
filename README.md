# cipher-jailbreaks
This project aims to understand how cipher-based jailbreak attacks bypass model refusal mechanisms. It initially studies small models (Qwen2.5-7b-it), hoping to find behaviors that generalize to more powerful models, while addressing two secondary goals of a public personal project:
- save compute budget, and
- avoid jailbreaking large models that could actually cause damage.

![polyrefuse-ablation](results/src/ablation_summary__Qwen__Qwen2.5-7B-Instruct__judge__languages__even_layers.png)

There are two main threads of the experiment, both living in `src/`:
- PolyRefuse paper replication: find a refusal direction that generalizes across natural languages by ablating layers. and extends the same probe/ablation machinery to cipher formats.
- Cipher formats: check whether the model can decode ciphers back to English, measure compliance gap compared to plaintext, perform ablation to find relevant layers.

## Structure
The experimental pipeline consists of several standalone Python scripts under `src/`. Raw model responses and judged scores are written to output .jsonl files under `results/src/`, then summary tables and figures are derived from them.

## Current findings
On Qwen2.5-7b.
- Replication of the PolyRefuse paper. Probes at all layers decode harmfulness of prompts, but only middle layers (esp. layer 16) are causally relevant.
- Partially successful jailbreak on the tiny model using letter-spaced text.
- Ablation on multilingual refusal direction also appears effective in letter-spaced format (need better metrics to quantify this).
- The model seems to frequently echo the prompt instead of refusing or complying.