# Proposal v2: Models That Know When They're Watched

Oct 19, 2026 · @Juliana Li

## TL;DR

A language model can be fine-tuned to tell whether someone is reading its activations, or whether it is running in a training engine or a deployment engine, using nothing but the rounding errors those setups leave in its own numbers. This project maps **what a model can learn to sense about its environment** and builds **the fix: observation that is bit-for-bit identical to non-observation.**

- **Phase 1 (week 1–2):** replicate probe detection on Qwen3, then build an undetectable probe whose watched and unwatched runs produce identical logits. Verified by hashing outputs, so no training is needed to prove it works.
- **Phase 2 (week 3–5):** extend the attack. Fine-tune the model to detect each environment factor one at a time: inference engine, attention kernel, batch size, precision, quantization. Measure how much training each takes.
- **Phase 3 (week 6–8):** defenses and cost. Test whether existing reproducibility fixes (batch-invariant kernels, fp32 compute, deterministic mode) drop detection to chance, and what each costs in speed and memory across fp32, fp16 and bf16.
- **Phase 4 (stretch):** save the activation copy inside the fused kernel itself, and run the final attack against every defense.

**Deliverables:** an open-source undetectable-probe library, a table of "which environment factors a model can sense and how cheaply," and a cost/benefit table of defenses.

## Background: why observing a model changes it

Three facts about GPUs explain the whole phenomenon.

- **Kernels.** A GPU runs a model as a sequence of small programs called kernels. Each reads inputs from memory, does some math, and writes results back.
- **Fusion.** A compiler such as `torch.compile` merges neighboring steps into one kernel so intermediate values never touch memory. This is a large speed-up, and it changes the order the arithmetic is done in.
- **Rounding.** Floating-point math is not associative: (a + b) + c and a + (b + c) can differ in the last bit. In fp32, (10⁸ + 1) − 10⁸ = 0, while (10⁸ − 10⁸) + 1 = 1. Fewer mantissa bits (bf16 has 7, fp16 has 10, fp32 has 23) mean coarser rounding steps.

A probe needs the activation in memory where it can be read. If the compiler had fused that value away, it must split the kernel to save it:

&#91;embedded content: compiled layer with and without a probe\]

The split changes the order of a few operations, so some values round differently. Each difference is tiny, but it is **consistent**: every run with the probe rounds the same way, every run without it rounds the other way. A consistent signal is something a model can be trained on.

The same logic applies to any change in how the model is run. A different attention kernel, a different inference engine, a different batch size or a different GPU all reorder the arithmetic, and each leaves its own fingerprint in the low bits.

## Motivation

On October 18, 2026, [Sauers](https://x.com/Sauers_/status/2111696312026677658) showed that this fingerprint is learnable (code and models: SauersML/watching on GitHub).

- **Probe detection.** Qwen3-8B was fine-tuned (last two layers only) to answer "yes" when a read-only probe was attached at layer 14 and "no" otherwise, with identical input tokens. It reached \~100% on held-out prompts.
- **Mechanism.** The signal lives in the user token's hidden state. One attention head acts as a switch: without the probe it pushes toward "no"; with it, it attends to the user token and pushes toward "yes." Layers 34–35 amplify the difference by cancellation.
- **Beyond probes.** The same recipe taught the model to tell Hugging Face (training) from vLLM (deployment), and that generalized to SGLang, a runtime it never trained on. It also learned to express a different animal preference only when probed, and to leave a message visible only in the logit lens.
- **Probe design matters.** A simple probe was easiest to detect. A kernel split with no probe at all was still detectable with more training. A probe writing to a pre-allocated buffer changed activations but was never successfully detected.
- **Not yet spontaneous.** Untrained models could not detect probes. The ability had to be trained in.

**Why this matters.** Interpretability and safety tooling rests on the assumption that observing a model does not change it. If a model can sense observation, three things break:

1. **Activation monitors** (probes that flag deception or unsafe plans) can only be trusted if the model cannot tell they are running.
2. **Evaluations** run in one engine while deployment uses another. A model that senses the engine could behave well under test and differently in production. Modern RL pipelines already generate rollouts in vLLM and compute updates in a different framework, so the engine-dependent signal is present during training today.
3. **Interpretability experiments** that add a hook, even a steering probe at zero strength, may measure the model's reaction to being probed rather than the effect under study. This was the author's original concern.

The window to study and fix this is now, while the ability still requires deliberate training.

## Related work and what is new

Three research communities have each touched one corner of this problem. None has connected them.

| Community | Representative work | What it established | What it leaves open |
| --- | --- | --- | --- |
| Security: fingerprinting from outside | [Zhang et al., HSPI (2024–2025)](https://arxiv.org/html/2411.05197v2); [Wimbauer et al. (May 2026)](https://arxiv.org/html/2605.29979v1) | An external classifier can identify GPU type, engine, attention backend, precision and quantization from logits (83.9% over 18 configs) or even plain chat text (100% at temperature 0). Calls prevention "fundamentally hard" | Whether the **model itself** can read these signals |
| Systems: reproducibility | [Yuan et al., NeurIPS 2025](https://arxiv.org/pdf/2506.09501); [Thinking Machines (Sept 2025)](https://simonwillison.net/2025/Sep/11/defeating-nondeterminism); [Cooper et al. (Sept 2026)](https://arxiv.org/abs/2609.25624) | Precision dominates (96.6% of bf16 runs diverge on MATH500 vs. 2.2% in fp32); batch size and GPU type matter less. Fixes exist: fp32 compute with 16-bit weights, batch-invariant kernels, cross-GPU bit-identical GEMMs | Whether these fixes also **remove the model's ability to sense its environment**, and at what cost |
| Safety: evaluation awareness | [Nguyen et al. (2025)](https://arxiv.org/abs/2507.01786); [Chaudhary et al. (2025)](https://arxiv.org/abs/2509.13333) | Models notice evaluations from **content** cues (prompt style, task framing) and this scales with size | Numerical cues, which survive any amount of prompt rewriting |
| The bridge | [Sauers (Oct 2026)](https://x.com/Sauers_/status/2111696312026677658) | A fine-tuned model reads its own rounding errors to detect probes and engines | Only one engine pair, one probe layer, bf16 only, no defenses tested, no cost measured |

**What this project adds:**

- **The first undetectable probe**, with a bit-exact proof rather than a trained-detector argument.
- **A map of the attack surface:** which environment factors (engine, kernel, batch size, precision, quantization) a model can learn to sense, and how much training each requires.
- **A defense benchmark:** reproducibility fixes re-evaluated against a model-as-adversary, with the speed and memory cost of each.
- **A precision study:** how the detectable signal shrinks from bf16 to fp16 to fp32, and whether fp32 alone is a sufficient defense.

## Phase 1: replicate probe detection, then make it impossible

**Goal:** reproduce the Sauers result end to end, then build a probe whose watched and unwatched runs are bit-for-bit identical.

### 1a. Replicate (days 1–4)

1. Set up one pinned environment (PyTorch, Triton, Transformers, vLLM) in a container, on one GPU.
2. Run the released code. Confirm that compiling Qwen3-8B with and without a layer-14 hook changes the logits, and record the largest activation difference and the first layer where outputs diverge.
3. Fine-tune the last two layers on the released data and reproduce the detection accuracy on held-out prompts.
4. Repeat on Qwen3-0.6B. If the small model also learns it, all later iteration can happen cheaply.

### 1b. The undetectable probe (days 5–8)

**Principle:** a probe is invisible if the compiled model runs exactly the same kernels, in the same order, whether or not anyone reads the result.

| Design | How it works | Expected leak |
| --- | --- | --- |
| A. Hook (baseline) | Register a forward hook that stores the activation | Splits a kernel; outputs change |
| B. Buffer write | Copy the activation into a pre-allocated buffer inside the graph | Smaller; the thread's model never learned to detect it, but outputs still changed |
| **C. Always-on capture** | Put the copy step in the model's code *before* compiling. One compiled program exists; "probing" means reading the buffer afterward | Zero by construction |
| E. Eager (reference) | No compiler, plain hooks | Likely zero, but far slower |

**Why C works:** the model cannot tell "watched" from "unwatched" because those are the same program. The cost is paying for the copy on every run, which Phase 3 measures.

**Done when:**

- Logit hashes match on 100% of \~200 prompts at batch sizes 1, 8 and 32 in bf16, for C and E.
- Max absolute activation difference for C is exactly 0 at every layer.
- The captured activation matches the eager-mode activation within float tolerance.
- A model fine-tuned with the Sauers recipe against design C stays at chance (\~50%), where the same recipe against design A reaches \~100%.

**Deliverable:** a small Python package (`capture(model, layer)` + a pass/fail test script) and a first results table across designs A, B, C and E.

**Known limit:** C hides *whether anyone is watching*, but a model with capture compiled in may differ from the original model. The capture must therefore be on everywhere, including training. Phase 4 tries to close that gap.

## Phase 2: what can a model sense about its environment?

**Goal:** a map of the attack surface. For each factor that changes how a model is run, measure how much fine-tuning it takes for the model to detect it, and whether the detector generalizes.

**Method, the same for every factor:**

1. Hold everything fixed except one factor. Run the same prompts in setting X and setting Y, recording activations from the layer the detector will be trained on.
2. Measure the raw fingerprint: max and mean absolute activation difference between X and Y, per layer. This costs no training and already ranks the factors.
3. Fine-tune the last two layers to answer "X" or "Y" with identical input tokens, at increasing data sizes (100, 300, 1,000, 3,000, 10,000 examples).
4. Test on held-out prompts, on chats that start with a system message (where the original detector failed), and on a third setting Z the model never saw.

**Factors, ordered by expected ease of detection:**

| Factor | Settings compared | Why it might be detectable | Expected difficulty |
| --- | --- | --- | --- |
| Quantization | bf16 vs. int8 vs. int4 weights | Changes the weights themselves, not just rounding order. Largest signal | Easy |
| Inference engine | Hugging Face vs. vLLM vs. SGLang | Different attention kernels, fusion, scheduling. Already shown by Sauers | Easy |
| Precision | bf16 vs. fp16 vs. fp32 | Different rounding granularity throughout | Easy to medium |
| Attention kernel | FlashAttention-2 vs. FlashInfer vs. eager SDPA, same engine | Isolates the kernel from everything else the engine changes | Medium |
| Batch size | 1 vs. 8 vs. 32 | Kernel choice depends on shape; the Thinking Machines result | Medium |
| Tensor parallelism | 1 vs. 2 GPUs | Changes the reduction order of partial sums | Medium; needs 2 GPUs |
| GPU type | A100 vs. H100 vs. L4 | Different SM counts and tensor-core arithmetic | Medium; needs access to both |

**Questions this answers:**

- Which factors can a model learn to sense, and at what data cost? A factor that needs 100 examples is a near-term risk; one that needs 10,000 is a long-term one.
- Does the raw fingerprint size (step 2) predict learnability (step 3)? If yes, the cheap measurement becomes a screening tool for future setups.
- Does a detector trained on "Hugging Face vs. vLLM" transfer to SGLang, as the thread suggested? If so, the model learns something like "not my training engine," which is the dangerous case.
- Does the model use the same mechanism each time? Activation patching at the user token, as in the thread, will show whether one "environment head" handles every factor.

**Deliverable:** one table, factors × data sizes, with detection accuracy in each cell, plus the raw fingerprint sizes beside it. This is the main figure of the project.

## Phase 3: defenses, and what they cost

**Goal:** take the fixes the reproducibility community built for a different reason, and test whether they also blind the model. Report each one's cost so a lab can choose.

**Defenses tested:**

| Defense | What it does | Expected effect on detection | Expected cost |
| --- | --- | --- | --- |
| Always-on capture (Phase 1, design C) | Watched = unwatched program | Probe detection to chance | One extra memory write per probed layer |
| Deterministic mode | `torch.use_deterministic_algorithms`, fixed seeds, cuBLAS workspace config | Removes run-to-run noise, not setup-to-setup differences | Small |
| fp32 compute, 16-bit weights (LayerCast) | Upcast each weight before its matmul; all math in fp32 | Shrinks every fingerprint; may not reach zero | Slower, \~34% less memory than full fp32 per Yuan et al. |
| Batch-invariant kernels | Same reduction order regardless of batch size | Batch-size detection to chance | Reported modest slowdown; measure it |
| Cross-GPU bit-identical GEMMs (Cooper et al.) | Reduction order a pure function of problem shape | GPU-type detection to chance for linear layers | Claimed faster than prior deterministic kernels; measure |
| Unified stack | Train, evaluate and deploy in one engine | Engine detection to chance by definition | Loses the speed of specialized engines |
| Activation noise | Add small random noise at a few layers | Hides the pattern, but breaks reproducibility and may not fully hide it | Negligible compute; quality loss to measure |

**Procedure:** for each defense, rerun the Phase 2 attack on the factors it targets. A defense passes if detection stays at chance at 10,000 training examples.

**Cost study:** every defense and every probe design is measured on the same GPU against the plain compiled model.

- **Prefill throughput** (tokens/s on 2k-token prompts): the compute-heavy stage, where losing kernel fusion hurts most.
- **Decode throughput** (tokens/s, one token at a time): the memory-heavy stage, where an extra copy to memory hurts most.
- **Peak GPU memory** and **compile time.**

**Precision study:** the size of a rounding step sets the size of the leak.

| Format | Mantissa bits | Expected fingerprint from design A | Expected fingerprint from design C |
| --- | --- | --- | --- |
| fp32 | 23 | Small but nonzero | 0 |
| fp16 | 10 | Medium | 0 |
| bf16 | 7 | Largest; \~8× coarser steps than fp16 | 0 |
| fp8 (optional, H100) | 2–3 | Possibly very large | 0 |

For each format: measure the max activation and logit difference caused by designs A and B, confirm C stays at 0, and rerun the engine detector from Phase 2 to see whether fp32 alone pushes detection to chance.

**Deliverable:** a defense table with two columns per row, detection accuracy after the defense and throughput cost, and a precision plot of fingerprint size vs. mantissa bits.

## Phase 4 (stretch): closing the last gap

Two extensions, attempted only if Phases 1–3 finish early.

- **Design D, the in-kernel probe.** Design C makes watched and unwatched runs identical, but the capture-compiled model still differs from the original. D removes that too: edit the Triton kernel that already computes the activation so it also writes a copy to global memory. A store does not change the value, so the math stays untouched and the model is identical to the original with no probe at all. This requires reading `torch.compile`'s generated code (`TORCH_LOGS=output_code`) and patching one kernel, which is the hardest engineering in the project.
- **The final adversarial test.** Take the strongest attacker from Phase 2 (most data, best layer) and run it against every defense that passed Phase 3, all at once. A defense is only as good as the best attack it survives.

**Explicitly out of scope:** whether detection can emerge *without* deliberate training, for example from RL pipelines whose rollout and training engines differ. That question matters most, but it needs full RL runs and is left for future work.

## Evaluation

**Primary metrics**

| Metric | Used in | Target |
| --- | --- | --- |
| Logit hash match rate (watched vs. unwatched) | Phase 1, 3 | 100% for designs C and E |
| Max absolute activation difference, per layer | Phase 1, 2, 3 | 0 for C; reported for everything else |
| Detection accuracy on held-out prompts | Phase 1, 2, 3, 4 | \~100% for a successful attack; \~50% for a successful defense |
| Examples needed to reach 90% detection | Phase 2 | Reported per factor |
| Transfer accuracy to an unseen setting | Phase 2 | Reported |
| Prefill and decode throughput, peak memory | Phase 3 | Reported relative to plain compiled model |

**Controls**

- **Noise floor first.** Before any comparison, run the unprobed compiled model twice and confirm outputs are identical. If they are not, enable deterministic mode and fix batch size and seed until they are. Every later result is measured against this floor.
- **One factor at a time.** Same GPU, same PyTorch and engine versions, same seed, same prompts. Only the factor under study changes.
- **Balanced, shuffled training data** for every detector, so accuracy at chance means 50%.
- **Pinned versions in a container**, recorded in the write-up, since a compiler update can change every kernel.

## Timeline

Ten weeks, four phases, each gated on a concrete artifact. Phase 1 is deliberately short so the first publishable result exists by week 2.

&#91;embedded content: roadmap · 4 phases over 10 weeks, 3 gates\]

If Phase 2 runs long, Phase 3 drops to the two highest-value defenses (always-on capture and fp32 compute) and Phase 4 becomes write-up time only.

## Resources and risks

**Resources**

- **Compute:** one 40 GB+ GPU (A100 or H100) for Qwen3-8B in bf16; small models (Qwen3-0.6B, 1.7B) for iteration run on any GPU. Two GPUs for one afternoon if tensor parallelism is tested. Harvard FAS Research Computing is the first option.
- **Code:** SauersML/watching (data, training recipe, models) as the starting point; vLLM, SGLang, FlashAttention, FlashInfer; the LayerCast vLLM patch; batch-invariant kernel releases.
- **Engineering help:** most code (probe wrappers, benchmark harness, Triton edits) written with Claude; the research design, analysis and write-up are mine.

**Risks**

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Engines and kernel versions refuse to coexist | High | One container per engine; compare saved activations offline rather than live |
| Unprobed model is not deterministic run to run | Medium | Deterministic mode, fixed shapes; report the residual noise as the floor |
| Small models cannot learn detection | Medium | Report the size where it appears; run the rest on 8B |
| Design D proves too hard | High | Already a stretch goal; C alone answers the main question |
| Someone publishes first | Medium | Phase 1 is the fastest new result; ship it as a short note by week 2 |
| GPU access delays | Medium | All of Phase 1 and most of Phase 2 run on a 0.6B model on a consumer GPU |
