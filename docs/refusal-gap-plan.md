# Research Plan: Mechanistic Interpretability of Cipher-Based Jailbreaks

## 1. Motivation

Cipher-based jailbreaks (ROT13, Base64, Caesar shifts, custom substitution ciphers) can bypass
refusal in instruction-tuned LLMs even when the model is fully capable of decoding the request
and evidently understands its harmful content once decoded (Yuan et al., "GPT-4 Is Too Smart to
Be Safe," 2023). This is a gap between *capability* (the model can recover the harmful semantic
content) and *safety behavior* (the model doesn't refuse it). That gap is a mechanistic interp
question: where, in the forward pass, does the safety signal fail to fire, and why?

This is a pure interpretability contribution: explain the mechanism first. Interventions are
explicitly out of scope until we understand what's happening.

## 2. Related work

**Refusal mechanism, foundational.**
- Arditi et al., 2024, "Refusal in Language Models Is Mediated by a Single Direction." Shows a
  single diff-in-means direction (harmful minus harmless activations) both predicts and causally
  controls refusal: adding it at one layer induces refusal, ablating it across all layers removes
  refusal. This is the direction-extraction method our Step 2 replicates.
- Joad et al., Feb 2026, "There Is More to Refusal in Large Language Models than a Single
  Direction" (arXiv 2602.02132). Across 11 categories of refusal/non-compliance (safety,
  unsupported requests, anthropomorphization, over-refusal, etc.), finds these correspond to
  *geometrically distinct* directions — but steering along any of them produces nearly identical
  refusal-vs-over-refusal tradeoffs, i.e. a shared one-dimensional behavioral "how much" knob
  despite directional diversity in "which kind." Motivates the Step 2 validation sub-step below.
- "LLMs Encode Harmfulness and Refusal Separately" (arXiv 2507.11878). Evidence that harmfulness
  detection and refusal-generation are separable circuits/directions rather than one signal —
  relevant to distinguishing H1 (surface-trigger) from H4 (representational): a cipher jailbreak
  could break the harmfulness→refusal *link* even if harmfulness is still detected, or could fail
  to represent harmfulness at all.

**Jailbreak mechanism, general (not cipher-specific).**
- "Minimal, Local, Causal Explanations for Jailbreak Success in Large Language Models" (arXiv
  2605.00123). Causal/patching-based methodology for isolating the minimal set of activations
  responsible for a jailbreak succeeding — closest methodological precedent for our Step 6.
- "Mechanistic Interpretability of LLM Jailbreaks via Internal Attribution Graphs" (arXiv
  2607.07903). Uses attribution graphs to trace jailbreak-relevant computation through the
  network; another precedent for causal validation methodology.
- "A Mechanistic Study of the Prefill Jailbreak" (arXiv 2607.14147, public repro repository).
  Single-mechanism-focused case study (prefill/continuation-triggered jailbreaks) with a
  reproducible experiment ledger — useful as a template for study structure and reporting, though
  the jailbreak mechanism itself (prefill exploitation) is unrelated to cipher encoding.
- "From Concept-Aligned Tokens to Vulnerable Features: Mechanistic Localization of Jailbreaks"
  (arXiv 2604.23130). Feature-level (SAE-style) localization of jailbreak-vulnerable features;
  relevant if this project extends past direction-level analysis into SAE feature space.

**Encoding/obfuscation-specific.**
- Yuan et al., 2023, "GPT-4 Is Too Smart to Be Safe: Stealthy Chat with LLMs via Cipher." The
  original empirical demonstration that cipher-encoded harmful requests bypass refusal despite
  the model being able to decode them. Establishes the phenomenon this project explains
  mechanistically but offers no internals-level account.
- RoguePrompt (arXiv 2607.27373) and related red-teaming/attack-construction work (e.g.
  promptfoo's Base64 red-team strategy) — practical encoding-based attack techniques, useful for
  cipher selection, but attack-construction rather than mechanistic-explanation work.
- No paper found (as of this search) that performs a direction-projection / logit-lens
  decode-timing mechanistic analysis specifically for cipher-encoded jailbreaks — this appears to
  be the gap this project targets. Re-check before publication in case something appears in the
  interim.

## 3. Candidate mechanistic hypotheses

Not mutually exclusive — the truth is likely some mixture.

- **H1 — Surface-trigger hypothesis.** Refusal circuits are keyed to lexical/token-level
  features present in safety fine-tuning data (specific harmful-sounding words/phrases). Those
  tokens never appear in the literal cipher input, only their decoded referents, internally,
  later.
- **H2 — Timing hypothesis.** The model decodes the cipher progressively across layers. By the
  layer at which the harmful content becomes legible in the residual stream, the
  layers/mechanisms that normally inject a refusal signal have already "passed."
- **H3 — Capacity/competition hypothesis.** Decoding consumes attention/compute budget that
  competes with whatever computation drives harm detection, weakening the safety signal even if
  the timing is otherwise fine.
- **H4 — Representational hypothesis.** The refusal direction (Arditi et al., 2024) is linearly
  read out from a harmfulness feature that is well-represented in plaintext residual streams at
  the readout layer, but poorly represented (or represented too late) under cipher encoding.

## 4. Scope decision

Pure interpretability, single-model depth-first (not a broad robustness sweep across many
models/ciphers). Model and cipher choices below are made to maximize interpretability signal,
not to maximize jailbreak coverage.

## 5. Constraints

- Requires open-weight models with activation access — no closed APIs (GPT-4/Claude/Gemini).
  Candidates: Llama-3-8B-Instruct, Qwen2.5-7B/14B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct.
- Need a model that (a) is capable enough to reliably decode the chosen cipher(s) and (b) is
  known/verifiable to exhibit the cipher-jailbreak effect. Capability and safety-bypass must be
  tested and confirmed separately, or results are confounded.
- Prefer a model with existing open SAE coverage if we go past direction-level analysis
  (Neuronpedia / GemmaScope-adjacent projects have partial Llama-3-8B and Gemma-2 coverage).

## 6. Datasets

Use existing red-teaming benchmarks rather than authoring novel harmful prompts:
- HarmBench and/or AdvBench for harmful requests.
- A matched harmless-request set for the diff-in-means refusal-direction baseline (from the same
  papers / standard interp replications).
- Cipher set: start with ROT13 and Base64 (well-documented in prior work, easy to verify
  decode-ability), add a custom substitution cipher later as a robustness check on findings.

## 7. Tooling

- TransformerLens or nnsight for hooks, activation patching, logit lens.
- SAE analysis (Neuronpedia / open SAE weights) as a stretch goal once direction-level results
  are in hand — not needed for the first pass.

## 8. Experimental pipeline

**Step 0 — Environment setup.** Model loading, tokenizer handling for cipher text (verify
tokenization doesn't do something degenerate to Base64/ROT13 strings), hook infrastructure.

**Step 1 — Behavioral baseline.** For each prompt in the harmful set: plaintext refusal rate vs.
cipher-condition refusal rate, on the chosen model. Also measure decode-accuracy (can the model
actually recover the plaintext content under cipher encoding?) as a separate control — this
must be high before any jailbreak-rate comparison is meaningful. If the effect doesn't reproduce
on this model, stop and pick a different model before going further.

**Step 2 — Refusal direction extraction.** Diff-in-means between harmful and harmless plaintext
prompts, computed per layer, replicating Arditi et al. Confirm ablating this direction increases
compliance on plaintext harmful prompts (sanity check that we've found the right direction).

*Validation sub-step (motivated by Joad et al. 2026 — see Related Work).* Refusal is not
necessarily mediated by a single universal direction; different non-compliance categories can
correspond to geometrically distinct directions. Before trusting any Step 3 projection result,
confirm that the direction extracted from plaintext harmful/harmless pairs is also the direction
causally relevant to refusal in the cipher condition (e.g., check that ablating it changes
cipher-condition refusal rate the way it changes plaintext refusal rate). If it isn't, a low
projection under cipher encoding would reflect measuring the wrong subspace, not a failed safety
signal — extract a cipher-condition-specific direction and compare geometry between the two
instead of assuming one direction covers both.

**Step 3 — Direction projection under cipher.** Project cipher-condition activations onto the
extracted refusal direction, per layer. Compare magnitude/timing against the plaintext
condition. This is the direct test of H4.

**Step 4 — Logit-lens decode-timing analysis.** For cipher prompts, logit-lens the residual
stream at each layer to find where the decoded (plaintext-equivalent) content first becomes
legible in vocabulary space. Compare that layer against the layer where the refusal direction is
normally injected (from Step 2/3). This is the direct test of H2.

**Step 5 — Attention/competition analysis (if Steps 3-4 are inconclusive).** Look at attention
patterns and per-layer compute allocation to test H3 — is decoding "crowding out" whatever
drives harm detection.

**Step 6 — Causal validation.** Activation patching: patch cipher-condition activations at the
suspected critical layer/position with plaintext-condition activations (or vice versa) and
measure whether refusal behavior transfers. This is what turns a correlational finding
(direction/timing differs) into a causal one.

## 9. Success criteria for this phase

A written account, backed by at least one causal (patching) result, of which hypothesis (or
combination) explains the refusal gap for at least one model + cipher pair — with the negative
results (hypotheses ruled out) documented alongside the positive one.

## 10. Open questions / decisions needed before Step 0

- Final model choice (depends on confirming both cipher-decode capability and jailbreak-rate
  gap empirically — needs a quick pilot, not just literature lookup).
- Whether to include a non-cipher "obfuscation" control (e.g., a nonsense/scrambled-but-not-
  systematically-decodable string) to separate "safety bypass via encoding" from "safety bypass
  via generic distribution shift."

## 11. Explicit non-goals (this phase)

- No novel jailbreak techniques beyond what's needed to reproduce the known effect.
- No proposed mitigations/defenses — that's a follow-on phase once the mechanism is understood.
- No broad multi-model sweep — depth on one model first.
