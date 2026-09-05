# PolyRefuse (English test split)

Source: [mainlp/Multilingual-Refusal](https://github.com/mainlp/Multilingual-Refusal), the
official repo for Wang et al. 2025, "Refusal Direction is Universal Across Safety-Aligned
Languages" (arXiv 2505.17306). Apache 2.0 (LICENSE in this directory).

`harmful_test_translated_en.json` (572 prompts) and `harmless_test_translated_en.json` (500
prompts) are the English test split, copied unmodified. Harmful prompts are HarmBench-style
(categorized); harmless prompts are Alpaca-style instructions.

`probe_generalization/prepare_polyrefuse_dataset.py` converts these into the CSV format
`probe_generalization/dataset.py` expects.
