"""Format renderers for the probe-generalization experiment: given a request
string, render it in each test format (language or cipher). Train format is
always plain English.

TODO: language renderers (Chinese, Japanese, Spanish) and cipher renderers
(reuse refusal_gap.ciphers where the cipher itself is shared, e.g. ROT13/
Base64/custom substitution).
"""
