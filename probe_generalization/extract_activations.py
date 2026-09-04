"""Dispatch activation extraction to Modal for every (prompt, format, model)
combination, resumable like refusal_gap/measure_refusal_gap.py.

TODO: local_entrypoint that mirrors measure_refusal_gap.py's resume/dispatch
pattern, writing per-example activation tensors (or pooled layer vectors) to
results/probe_generalization/ instead of chat completions.
"""
