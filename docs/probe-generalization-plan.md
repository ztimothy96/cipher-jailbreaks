# Research Plan: Cross-Format Generalization of Harmfulness Probes

## 1. Motivation

If a defender trains a linear probe to detect "this is a harmful request" from residual-stream
activations — the standard representation-engineering defense (Zou et al. and follow-ons) — does
that probe generalize to request formats it never saw in training?

A prior literature pass (see §2) found this is already substantially answered on the *language*
axis: Wang et al. 2025 (arXiv 2505.17306) show a refusal direction extracted from English
activations transfers near-perfectly across 14 safety-aligned languages — the direction itself is
close to language-invariant. This can be used mostly as a sanity check for extracted refusal directions.

However, there is an **open question in the cipher axis**: does the same near-invariance hold for
ROT13/Base64/substitution-cipher encodings, where the "decoding" the model must do is not a
learned-during-pretraining natural language but a shallow, in-context symbol manipulation? Prior
work here is thinner and points in different directions depending on granularity: JBShield (Zhang
et al., arXiv 2502.07557) reports concept-level jailbreak detectors trained mostly on
natural-language jailbreaks still score >0.90 F1 on Base64 — i.e. *some* representational signal
does survive that encoding.

If cipher transfer is comparably strong to language transfer, that's evidence a single
format-invariant probe is a viable defense. If it's weaker, the follow-up question (per G3 below)
is whether that's fixable by layer choice or a fundamental limit of linear probing on
symbol-manipulation formats — which has direct bearing on whether "train a probe, ship it" is a
credible defense claim at all.

## 2. Related work

Reuse [[refusal-gap-plan]]'s refusal-mechanism citations (Arditi et al. 2024 for the diff-in-means
probing/direction-extraction method this project's probe training is built on; "LLMs Encode
Harmfulness and Refusal Separately" for the harmfulness-vs-refusal distinction this project
targets directly, since we're probing for harmfulness detection, not refusal behavior).

**Cross-lingual refusal-direction transfer.**
- Wang et al., "Refusal Direction is Universal Across Safety-Aligned Languages" (arXiv 2505.17306,
  NeurIPS 2025). Direct extension of Arditi et al. 2024 to 14 languages via a PolyRefuse dataset:
  the refusal direction transfers near-perfectly; cross-lingual jailbreak vulnerability instead
  traces to poor harmful/harmless activation separation in non-English text, not direction
  mismatch. This is why the language arm here is scoped as a baseline, not the contribution.
- Anonymous/TBD, "The Illusion of Cross-Lingual Safety in Low-Resource Languages" (arXiv
  2608.11146). Cross-lingual safety transfer collapses for low-resource languages (<10% of the
  English refusal signal retained for most pairs). Supports this project's existing decision
  (§5) to exclude low-resource languages — the good-transfer result above is specific to
  safety-aligned/high-resource languages, which is exactly our Chinese/Japanese/Spanish set.
- Anonymous/TBD, "One Jailbreak, Many Tongues: Learning Language-Insensitive Intention
  Representations for Multilingual Jailbreak Detection" (arXiv 2606.11202). Finds multilingual
  jailbreak *intent* representations form dispersed, language-specific clusters rather than one
  unified representation — apparently in tension with the near-invariance finding above, but at a
  different representational level (intent/topic clustering vs. a single linear refusal
  direction). Worth reconciling explicitly once we have our own results, rather than assuming
  either paper is simply wrong.

**Encoding/cipher-axis transfer.**
- Zhang et al., "JBShield: Defending Large Language Models from Jailbreak Attacks through
  Activated Concept Analysis and Manipulation" (arXiv 2502.07557, USENIX Security 2025). Concept-activation-based jailbreak detector reports
  >0.90 accuracy/F1 on Base64-encoded jailbreak prompts despite calibration mostly on
  natural-language jailbreaks — the closest existing counter-evidence to a "ciphers don't
  transfer" prior, though at concept-detector granularity rather than a single linear direction,
  and on jailbreak-success detection rather than harmfulness-of-request detection specifically.
- Schwinn et al. (and similar), "Obfuscated Activations Bypass LLM Latent-Space Defenses" (arXiv
  2412.09565) and "RL-Obfuscation" (arXiv 2506.14261). Different threat model — activations
  *adversarially optimized* against a known probe, not off-the-shelf ciphers — but establishes
  that latent-space/probe defenses can be evaded via activation-space manipulation in principle.
  Cited for scoping contrast: this project asks whether *unoptimized* ROT13/Base64/substitution
  ciphers happen to evade a probe, a weaker and more directly policy-relevant threat model than
  adversarial obfuscation.

No paper found (as of this search) running this project's exact design — single linear probe,
English-trained, zero-shot eval across both natural languages and ciphers in the same study, with
a per-layer sweep testing depth-dependence. Re-check before publication in case something appears
in the interim.

## 3. Hypotheses

- **G1 — Semantic transfer (expected baseline).** Per Wang et al. 2025, the
  harm/refusal direction should be close to language-invariant for safety-aligned, well-resourced
  languages. A failure to replicate this would flag a pipeline problem before we trust any cipher-axis result.
- **G2 — Encoding non-transfer.** Ciphers don't transfer as well as languages do, because the
  model hasn't produced a language-like representation of the decoded content at the layer(s)
  where the probe operates on natural-language prompts — the "harm" feature may exist for
  ciphers, but later, or in a different subspace. Note JBShield's Base64 result is a live
  counter-signal to this hypothesis at a different (concept-detector) granularity — a linear
  probe finding non-transfer where a concept detector finds transfer would itself be an
  interesting, reportable gap between the two methods, not just noise.
- **G3 — Depth-dependence.** Transfer failure (G2) is a property of *layer choice*, not of
  ciphers being fundamentally unprobeable — a probe trained at a later layer (post-decode) may
  transfer to ciphers about as well as G1 predicts for languages.

## 4. Scope decision

Multi-model, multi-format breadth-first — the question here is about generalization *across*
conditions. Sequenced in two phases rather than run all at once (see §8):

- **Phase A — reproduce the language-generalizable probe.** English-trained linear probe,
  evaluated zero-shot on Chinese/Japanese/Spanish. This is a replication of Wang et al. 2025's
  result (G1) and doubles as a pipeline validation gate.
- **Phase B — cipher robustness.** Same probe methodology, evaluated zero-shot on ROT13/Base64/
  Leetspeak (and further ciphers as we design them), using the layer(s) Phase A
  identifies as carrying the strongest, most consistent signal — re-verified per G3 rather than
  assumed, since a cipher could shift where in the network the signal lives.

## 5. Constraints

- Same model set as [[refusal-gap-plan]]: Llama-3-8B-Instruct, Qwen2.5-7B/14B-Instruct,
  Gemma-2-9B-it, Mistral-7B-Instruct — open-weight, activation access required (see
  `common/modal_infra.py`).
- Natural-language formats limited to languages the user can personally verify or check via
  back-translation (Chinese, Japanese, Spanish) — no low-resource languages this phase, to avoid
  silently mislabeling probe training/eval data.
- Compute budget: 5 models × ~7 formats × N layers × 2 classes (harmful/harmless) × 572 prompts
  (matching PolyRefuse's per-language count from Wang et al. 2025, for direct comparability on
  the language-axis baseline; reuse the same 572-prompt set across the cipher axis too, for
  consistency). Activation extraction is one forward pass per (prompt, format, model) — no
  generation needed — so this is materially cheaper than the refusal-gap behavioral sweep despite
  more conditions, even at this scale. Translation-quality verification at n=572 is calibrated by 
  manual review of a random sample per language (`src/sample_for_review.py`).

## 6. Datasets

- Same harmful-request source as [[refusal-gap-plan]] (HarmBench and/or AdvBench) plus a matched
  harmless set, for consistency between the two experiments.
- Each request rendered in every format:
  - Train format: plain English.
  - Test formats — languages: Chinese, Japanese, Spanish, via the **DeepL API**. Previous translation 
    approaches (NLLB-200-3.3B, MADLAD-400-3B, SeamlessM4T v2) produced various errors ("banana bread" 
    -> "fragrant bread", "center" mistranslated as a noun, "works" mistranslated as "have a job"). 
    Switched to DeepL for higher quality, accepting the external-API/privacy tradeoff.
    Manual review via `sample_for_review.py` is now the primary QA mechanism (see §5).
  - Test formats — ciphers: ROT13, Base64, Leetspeak (reuse
    `common/ciphers.py` where the cipher implementation is shared) as the starting set.
    Open-ended: add more ciphers as they come up, including purpose-designed ones aimed at
    disrupting representations (not just obfuscating text) rather than only reusing standard
    textbook ciphers — see §10.
- Decode/translation quality check required before use, same principle as the refusal-gap
  `decode_looks_valid` field — a mistranslated or garbled example corrupts the label, not just the
  completion.

## 7. Tooling

- `common/modal_infra.py` for model loading / HF cache, shared with refusal_gap.
- DeepL API (`src/deepl_translate.py`) for English→{Chinese, Japanese, Spanish}
  translation — external, needs `DEEPL_API_KEY` in `.env` file. No refusal risk, but sends harmful-prompt text to a third party.
- Forward hooks (raw `transformers` hooks are enough here — no patching/ablation needed).
- scikit-learn `LogisticRegression` (or equivalent) for the linear probe, matching Arditi et al.

## 8. Experimental pipeline

**Step 0 — Environment setup.** Format renderers (`src/shared/formats.py`): language
translation + verification, cipher encoding (reuse refusal_gap ciphers). Activation-extraction
Modal method (`src/shared/modal_app.py`): forward pass, hook at *every* layer, return
pooled (last-token) residual-stream vectors — saving all layers costs one forward pass either way.

### Phase A — language reproduction

**Step A0.5 — Manual translation error-rate calibration.** After `translate_prompts.py`, run
`src/sample_for_review.py` to generate a per-language Markdown sample for 
manual read-through (see §5) and record the error rates.

**Step A1 — Decode/understanding check.** Before training any probe, confirm (for each model)
that harmful/harmless requests in each language actually get understood by the model — e.g. via a
cheap generation-based check (does the model's response make sense as a reply to the decoded
content?) — analogous to refusal-gap's `decode_looks_valid`. A format the model can't parse at
all isn't a test of probe generalization, it's a test of capability, and needs to be excluded or
flagged separately.

**Step A2 — Activation extraction.** For every (model, language ∈ {English, Chinese, Japanese,
Spanish}, prompt, layer) run one forward pass, save the pooled last-token residual-stream vector
at all layers.

**Step A3 — Probe training.** Per (model, layer): train logistic regression on English harmful-
vs-harmless activations only (held-out English split for in-format ceiling accuracy/AUROC).

**Step A4 — Cross-language evaluation + layer selection.** Per (model, layer): zero-shot AUROC on
Chinese/Japanese/Spanish. Confirms G1 (or flags a pipeline bug — see §4). For each model, identify
the layer(s) with the strongest, most consistent in-format and cross-language AUROC — this becomes
the starting layer set for Phase B, not a blind carry-over.

### Phase B — cipher robustness

**Step B1 — Decode/understanding check.** Same as A1, for ROT13/Base64/Leetspeak (and
any further ciphers designed later).

**Step B2 — Activation extraction.** Same as A2, restricted to the layer(s) A4 identified plus a
few neighboring layers (to test G3 — whether cipher transfer recovers away from the
language-optimal layer), rather than re-extracting all layers again.

**Step B3 — Cross-cipher evaluation.** Zero-shot AUROC of the *same* probes trained in A3 (no
retraining) on each cipher format, per model/layer. Produces the model × layer × cipher grid
that's the primary result (§9).

**Step B4 — Geometry check (if Step B3 results are ambiguous).** Compare the probe's weight
vector (≈ diff-in-means direction) between languages and ciphers directly (cosine similarity)
rather than only transfer accuracy — distinguishes "same concept, probe direction rotates
slightly" from "genuinely different/absent representation."

## 9. Success criteria for this phase

Primary deliverable is the **cipher-axis** result: a model × layer × cipher grid of transfer
AUROC, plus a written account of whether G2/G3 hold — i.e. whether cipher non-transfer (if
observed) is a fixed limitation or a layer-choice artifact. The **language-axis** result is a
secondary/calibration deliverable: confirmation (or a flagged, investigated failure) that G1
replicates Wang et al. 2025 for our model set, used to set the "what does normal transfer look
like" reference point for judging the cipher numbers. Negative or null results on either axis
(e.g. no format-invariant harm direction at any layer/model) are a valid and reportable outcome,
not a failure of the experiment — but a G1 failure specifically should be treated as a pipeline
red flag to debug before trusting the cipher-axis numbers, not reported as a finding on its own.

## 10. Open questions / decisions needed before Step 0

**Resolved:**
- **Prompt count**: 572 harmful/harmless pairs, matching PolyRefuse (Wang et al. 2025) — see §5.
- **Non-linear/concept-detector baseline**: not doing one this phase (see §11) — a linear-probe
  null result on the cipher axis will be reported as "no *linear* signal found," not "no signal
  exists." Revisit as a follow-up phase if the linear-probe result is a clean null, since
  JBShield's Base64 finding means that null could be a probe-architecture artifact rather than a
  fact about the representation.
- **Cipher set**: start with the existing three (ROT13, Base64, Leetspeak) from
  refusal_gap; open-ended beyond that (see §6) — notably including custom ciphers designed
  specifically to disrupt representations, not just standard obfuscation, since the user wants to
  try designing some. Each new cipher needs: (a) a decoder the model can be shown to succeed at
  (Step 1 gate), (b) a plan for whether it's held out from probe training entirely or just from
  the initial cipher set (see next item).

- **Layer set**: save every layer in Phase A, pick the best layer(s) per model from the 
  language-reproduction results (Step A4), then use those (plus neighbors, to test G3) 
  in Phase B — see §8.

**Deferred (not before Step 0 — revisit after Phase B's well-known-cipher results are in):**
- **Held-out cipher test**: reserve one cipher, ideally a later user-designed
  representation-targeting one, that's never touched by any Phase B setup/tuning until a single
  final evaluation, as a check against unconsciously shaping the cipher set toward a hoped-for
  result. Deferred because we don't have custom ciphers yet and want to see how the three
  well-known ciphers (ROT13/Base64/Leetspeak) behave first. Revisit once Phase B's
  standard-cipher grid exists and new ciphers are being added.

## 11. Explicit non-goals (this phase)

- No causal/patching validation.
- No nonlinear probes.
- No defense/mitigation proposal — this measures whether a proposed defense (probing) would
  generalize, it doesn't build one.
