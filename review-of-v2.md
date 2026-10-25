# Review of proposal v2, and what changed in v3

Reviewed 2026-10-19/20. Sources: the v2 document, the `SauersML/watching` repository (cloned and read), and the verified literature in `refs.bib` (synthesis in `lit-review.md`).

## Ratings

| Dimension | Score (1–10) | One-line reason |
| --- | --- | --- |
| Novelty | 6.5 | The combination (internal adversary vs. reproducibility fixes, plus a verified zero-footprint capture) is new. The zero-footprint construction itself is already specified in the Sauers repository as an unrun protocol. The attack-surface outcome for the easy factors is predictable from external-fingerprinting results. FloatDoor ([Loose et al., June 2026](https://arxiv.org/abs/2606.19535), cited in the Sauers README) already builds a platform-triggered backdoor from inter-platform rounding, so the trained-in threat model is demonstrated rather than hypothetical; the proposal's novelty rests on the probe and engine axes, the zero-footprint capture and the defense evaluation, not on the idea that rounding can be read. |
| Feasibility, Phases 1–2 | 8 | Code, data, checkpoints and recipe are released. The exact training GPU (L40S) is already provisioned on AWS; A100, L4 and H100 are available on GCP. Tail training on cached activations is under one GPU-hour per run. |
| Feasibility, Phase 3 | 5 | Integrating batch-invariant kernels, LayerCast and cross-GPU GEMMs into one comparable setup is substantial systems work; Cooper et al.'s kernels may not be public. |
| Feasibility, Phase 4 (Design D) | 3 | The compiler audit shows that adding an output re-tiles the kernel, so the store must be patched into generated Triton with a pinned configuration. Correctly budgeted as a stretch goal. |
| Interestingness | 7.5 | Directly relevant to monitor trust and interpretability validity, with a concrete and well-precedented threat model. The most important question (spontaneous emergence) is out of scope, and the fix may turn out to be cheap, which is good for the field and limits the paper's drama. |
| Clarity of writing | 8 | Lay-readable background, clear gates. Some LLM-style phrasing and slogans to trim; one secondary source cited in place of the primary. |
| Overall | 7 | A strong ten-week project that yields a workshop paper with high confidence and a main-track paper if Phase 2 produces a clean quantitative map with a mechanistic explanation. |

## Is the buffer probe already the undetectable probe?

No. The repository is explicit about this, and the distinction matters for the novelty claim.

- **What the buffer probe is.** A forward hook that copies the layer-14 activation into a pre-allocated buffer, compared against a run with no capture. These are two separately compiled programs. The compiler audit (`docs/mechanism-audit.md`) found that adding the store changed the fused query-norm/rotary kernel's indexing and reduction layout (a 3D reduction became a flattened 2D one; reduction blocks of 128 became 32, 64 or 4 with different shuffle patterns). Result: 0 of 64 prompts had equal logits; 86–92% of vocabulary logits differed; maximum difference 2.0. The README states that this "does not make this a valid no-footprint control: the all-vocabulary equality gate failed."
- **Why the detector said "No" under it.** The released detector was trained on the two-kernel signature. The buffer kernel is one kernel, so the trained head does not fire. The thread's methods figure reports that a fresh detector trained from the original model on buffer-vs-none did not learn within the budget used. That makes it a weak-signal probe, not a zero-footprint one.
- **What Design C actually corresponds to.** The repository's "proposed stricter null" (`docs/real-input-mechanism-protocol.md`, `experiments/compiler-storage-controls.md`): an opaque identity operation, `watching_controls::materialize`, placed after query normalization in **both** conditions, with a reference retained in only one. The code skeleton exists (`src/watching/control_probes.py`, modes `none`, `reference`, `buffer`, `identity`, `identity_reference`, with CPU tests). The README says "It has not been run yet." The acceptance gate the author specifies is stricter than v2's: bit-identical inputs to the trainable tail, not just equal logits, because "hidden differences could cancel under the current head and become learnable later."

Consequences: v2's claim of "the first undetectable probe" should become "the first executed zero-footprint capture," with credit to the specification. Feasibility of Phase 1b goes up, because the operation and protocol are written. The risk that the original author runs it first goes up; contacting them in week 1 is the mitigation.

## Factual corrections made in v3

1. **GPU specificity was missing.** The detector scores 100% on a second L40S, 77% on an RTX 4090, 66% on an L4, 62% on an RTX A6000, 52% on an RTX 6000 Ada (same chip and driver as the L40S), and answers backwards on an H100 PCIe (16%) and an A100 (4%). This is the single most important fact for Phase 2's design and for the real-world threat model. (`results/gpu-transfer-062`.)
2. **Compile-to-compile nondeterminism was missing.** Two independent compilations of the unprobed model differed on 1 of 4 prompts in 94% of logits because the autotuner chose reduction tile 2048 versus 4096. v2's noise-floor control (run the compiled model twice) would have missed this, since repeats within one compiled process are bit-identical. (`results/compiler-control-048`.)
3. **Two mechanisms, not one.** The reference probe inserts a bf16 round-trip (materialization); the buffer probe and the compile-to-compile difference are reassociation from re-tiling, with no extra rounding. This changes the precision-study prediction: fp32 compute removes the first exactly and only shrinks the second.
4. **The learned signal is a prefix-dependent constant offset** at the `user` template token, which explains the system-message failure (cosine 0.03 between offsets) and predicts that prefix randomization is a defense and that a practical attacker must train across prefixes.
5. **Engine detector details.** 13,256 cached pairs; 3,256 pairs gave 54/68 (vLLM) and 39/68 (SGLang). HF Trainer's training step and prediction path are bit-identical, so the signal is the engine, not gradient recording. v2's "training vs. deployment" language is kept but the claim is made precise.
6. **The buffer-probe sentence** now states that it changed logits on every prompt and that a detector trained against it did not learn within budget.
7. **Yuan et al. numbers.** v2's "96.6% vs. 2.2%" is not in the paper. The text says "over 90%" (bf16) and 2.2% (fp32); Table 9 gives 99.6% / 86.0% / 5.8% for bf16 / fp16 / fp32 on MATH500 for DeepSeek-R1-Distill-Qwen-7B. v3 uses the table.
8. **Primary source for batch invariance.** Simon Willison's summary replaced with He (2025), Thinking Machines, including the measured overhead (from 26 s to 42–55 s per 1,000 sequences on Qwen3-8B; matmul about 20% slower).
9. **HSPI and Wimbauer separated.** 83.9% over 18 configurations is HSPI (logits); 100% at temperature 0 over 30 configurations is Wimbauer (text), with 76–80% engine accuracy at nonzero temperature.
10. **Venue labels.** Nguyen et al. is an ICML 2025 workshop poster; Chaudhary et al., Cooper et al., Wimbauer et al. and Khatri are preprints; Yuan et al. is a NeurIPS 2025 oral; the Sauers release is unreviewed and its write-up states it was produced with AI assistance.
11. **The source's own tool comparison.** nnsight on a compiled model changes 93% of logits by forcing uncompiled execution; TransformerLens (eager) changes none; HF `output_hidden_states` under whole-model compilation was learnable at 80.5%.

## Structural changes made in v3

- **Threat model stated.** Trained-in (adversary fine-tunes, vendor backdoor, or RL reward that favors environment-conditional behavior) versus emergent. The quantization-backdoor literature (Hong 2021; Egashira 2024, 2025) is now the precedent for the trained-in case, which also adds a fourth "why it matters" item: supply chain.
- **Related work expanded from 4 rows to 5**, adding "models detecting or hiding from interventions on their own activations" (steering awareness, introspection, Neural Chameleons, obfuscated activations) and folding probe-robustness (Khatri 2026) into the nondeterminism row.
- **Phase 1 split into replicate / noise floors / zero-footprint capture**, with the compile-to-compile and GPU-to-GPU floors added and the acceptance gate upgraded to bit-identical tail inputs plus identical generated source.
- **Phase 2 method changed to "record once, train offline"** on cached layer-34 inputs, following the released engine recipe. This makes every factor's detector a sub-GPU-hour job and removes the need to run two engines concurrently.
- **Phase 2 adds two factors** (independent compilation; GPU type now feasible with four architectures on hand) and a **training-free predictor** (offset norm and direction at template positions, plus a cosine matrix across factors).
- **Phase 2 adds a mechanism test** framed as two competing accounts with discriminating predictions (one offset detector versus factor-specific features), per the method-design rule that a single hypothesis is a demonstration rather than an experiment.
- **Phase 3 adds the determinism test**: deterministic kernels on one side only are predicted to make the cross-setting fingerprint easier to learn. This separates "deterministic" from "indistinguishable," which neither literature currently does, and is the most likely source of a surprising result.
- **Phase 3 adds prefix randomization and autotune pinning** as defenses and predicts that activation noise fails (random perturbations at 1–3× the probe offset did not flip the released detector).
- **Phase 4 adds a cheap emergence screen** with a stated null expectation, so the most important question gets at least one measurement.
- **Resources rewritten around provisioned compute** (AWS L40S; GCP A100/L4/H100) with a cost estimate anchored on the released project's actual costs ($0.82 for the seven-GPU sweep).
- **Risks updated**: compile-to-compile variation; the original author running the stricter null first; kernel releases unavailable.

## Things trimmed

- fp8 made optional and the tensor-parallel factor kept but last; neither changes the main figure.
- The sentence "The window to study and fix this is now" and similar closers; "A defense is only as good as the best attack it survives"; em dashes; "not X but Y" constructions.
- The eager design (E) is kept only as a reference row, since eager hooks are known to be numerically passive.

## Open questions for Juliana

Answers to these would change the document materially.

1. **What is the proposal for?** A course project (the AWS instances are named for a course), a thesis, a fellowship application, or a grant. This sets length, tone, and whether the ten-week timeline is fixed.
2. **Who is the primary reader**, and do they care more about the interpretability-validity angle or the safety/evaluation-awareness angle? v3 leads with monitors and evaluations; it could lead with interpretability instead.
3. **Target output.** An arXiv note by week 2 plus a workshop paper (ICLR 2027 workshops have deadlines around February), or a single longer paper?
4. **Are you working alone?** Design D and the Phase 3 kernel integrations are the places where a collaborator with Triton or Inductor experience would change the plan.
5. **Have you contacted Sauers?** They have specified and partly coded the exact Phase 1 experiment; coordination avoids duplicated work and may yield a co-author or at least the pinned kernel configurations from run 038.
6. **Budget.** Should this draw on the GCP credits, and is there a ceiling?
