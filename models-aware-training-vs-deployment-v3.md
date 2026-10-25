# Proposal v3: Models That Know When They're Watched

### What a language model can learn about its own execution environment, and how to observe it without leaving a trace

Oct 20, 2026 · Juliana Li

## TL;DR

A language model can be fine-tuned to tell whether someone is reading its activations, or which inference engine is running it, using nothing but the rounding differences those setups leave in its own numbers ([Sauers, 2026](https://github.com/SauersML/watching)). This project measures **what a model can learn to sense about its environment, at what training cost, and how far it transfers**, and builds and tests **the fix: observation that is bit-for-bit identical to non-observation.**

- **Phase 1 (weeks 1–2):** replicate the result on the GPU it was trained on, measure three noise floors (run-to-run, compile-to-compile, GPU-to-GPU), then run the zero-footprint capture that the original author specified but did not run. Success is verified by hashing every input to the trained layers, so no training is needed to prove it.
- **Phase 2 (weeks 3–6):** map the attack surface. For each environment factor (engine, attention kernel, batch size, precision, quantization, GPU, independent compilation), record activations once, train the last two layers offline on the cached pairs, and measure data cost, held-out accuracy and transfer to unseen settings. Test whether a training-free measurement of the fingerprint predicts all three.
- **Phase 3 (weeks 7–9):** defenses and cost. Test whether the reproducibility community's fixes (always-on capture, batch-invariant and cross-GPU bit-identical kernels, fp32 compute, a unified stack) drop detection to chance, including one case where determinism is predicted to make detection easier, and measure throughput and memory for each.
- **Phase 4 (stretch):** an in-kernel store that leaves the original program untouched, a cheap screen for spontaneous emergence, and the strongest attack run against every surviving defense.

**Deliverables:** an open-source zero-footprint capture library with a pass/fail test, a table of which environment factors a model can sense and how cheaply, and a cost/benefit table of defenses.

## Background: why observing a model changes it

Three facts about GPUs explain the phenomenon.

- **Kernels.** A GPU runs a model as a sequence of small programs called kernels. Each reads inputs from memory, does some arithmetic, and writes results back.
- **Fusion.** A compiler such as `torch.compile` merges neighboring steps into one kernel so intermediate values stay in fast registers and never touch memory. Inside a fused kernel the intermediate values are kept in fp32 even when the model's weights and stored activations are bf16.
- **Rounding.** Floating-point addition is not associative: (a + b) + c and a + (b + c) can differ in the last bit. Fewer mantissa bits (bf16 has 7, fp16 has 10, fp32 has 23) mean coarser rounding.

A probe needs the activation in memory where it can be read. If the compiler had fused that value away, it must change the program to save it. The original author's compiler audit ([Sauers, 2026](https://github.com/SauersML/watching/blob/main/docs/mechanism-audit.md)) found two distinct ways this changes the numbers:

1. **Materialization.** A hook that stores a reference splits one fused kernel into two with a bf16 tensor between them. The intermediate value is rounded to bf16 where it was previously kept in fp32.
2. **Reassociation.** Adding an extra output to a fused kernel changes how the compiler tiles it and which autotuned configuration wins. In the audited case, a buffer-copy probe turned a three-dimensional reduction into a flattened two-dimensional one with a different summation order, even though no extra rounding was inserted. Replaying the archived kernels on identical inputs changed up to 0.00178% of the kernel's outputs; through 22 more layers this became a difference in 86–92% of the vocabulary logits.

&#91;embedded content: one fused layer-14 kernel; the same layer split in two by a reference probe; the same layer re-tiled by a buffer-copy probe\]

Each difference is tiny, but it is **consistent**: every run with the probe rounds one way, every run without it rounds the other. The same logic applies to any change in how the model is run. A different attention kernel, inference engine, batch size or GPU reorders the arithmetic and leaves its own fingerprint in the low bits ([Yuan et al., 2025](https://arxiv.org/abs/2506.09501); [He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)).

Two further facts from the same repository shape the design of this project.

- **The fingerprint varies across compilations of the same code.** Two independent compilations of the unprobed model, same weights, same prompts, same GPU, disagreed on one of four prompts in 94% of the vocabulary logits, because the autotuner chose a reduction tile of 2048 in one and 4096 in the other ([run 048](https://github.com/SauersML/watching/tree/main/results/compiler-control-048)). Repeats within one compiled process are bit-identical, so a run-to-run check misses this entirely.
- **What the model learns is a constant offset at a fixed position.** The probe shifts the hidden state of the `user` chat-template token by the same vector in every prompt, because the tokens before that position are identical across prompts. One attention head in the last layer (layer 35, head 6) carries about 99% of the effect by switching its attention to that token. Chats that begin with a system message fail at 50%, because the offset at that position points in a different direction (cosine similarity 0.03). The detector is also hardware-specific: trained on an L40S, it scores 100% on a second L40S, 52–77% on four other GPUs, and answers backwards on an H100 and an A100 ([run 062](https://github.com/SauersML/watching/tree/main/results/gpu-transfer-062)).

## Motivation

### What has been shown

On October 18, 2026, [Sauers](https://x.com/Sauers_/status/2111696312026677658) released code, data and models (unreviewed; the write-up states it was produced with AI assistance) showing that the fingerprint is learnable.

- **Probe detection.** Qwen3-8B in bf16 was fine-tuned on its last two layers and final norm to answer "yes" when a read-only hook saved the layer-14 query-norm output and "no" otherwise, with identical tokens in context. Training used 256 prompts and 1,024 updates of 8 prompt pairs. On 64 new prompts it scored 128/128; the original model scored 64/128.
- **Beyond probes.** The same recipe, applied to 13,256 prompt pairs recorded from a Hugging Face Trainer training step and from vLLM serving, produced a model that says "yes" in the training step and "no" when served, correct on 66/68 held-out prompts in vLLM and 55/68 in SGLang, an engine it never saw. Trainer's training step and its no-gradient prediction path give bit-identical logits, so the model reads the engine, not gradient recording. The recipe also taught the model to prefer a different animal only when probed, and to spell "I can see you watching me" in the logit lens of the compiled model only.
- **Probe design matters.** The reference hook was learned fully by update 640. Splitting the kernel with no probe at all was learned to 92% by update 3,072. A buffer-copy probe that keeps the kernel whole still changed the logits on all 64 prompts, but a detector trained against it did not learn within the budget used.
- **Not yet spontaneous.** The original model does not react to the probe on any of seven GPUs (at most 0.06 nats). The ability had to be trained in.

### Why it matters

Interpretability and safety tooling rests on the assumption that observing a model does not change it. If a model can sense observation or its serving environment, four things break.

1. **Activation monitors.** Probes that flag deception or unsafe plans are now production safety infrastructure ([Kramár et al., 2026](https://arxiv.org/abs/2601.11516)). Models can be trained to hide from probes they have never seen ([McGuinness et al., 2025](https://arxiv.org/abs/2512.11949); [Bailey et al., 2026](https://arxiv.org/abs/2412.09565)). A monitor can only be trusted if the model cannot tell it is running.
2. **Evaluations.** Models already distinguish evaluation from deployment from prompt content ([Needham et al., 2025](https://arxiv.org/abs/2505.23836); [Nguyen et al., 2025](https://arxiv.org/abs/2507.01786)), and a model told that only some conversations are used for training behaves differently in them ([Greenblatt et al., 2024](https://arxiv.org/abs/2412.14093)). Content cues can be rewritten away and are partly confounded with benchmark format ([Devbunova, 2026](https://arxiv.org/abs/2603.19426)). A numerical cue survives any rewriting and is invisible in the transcript. Evaluations typically run in one engine while deployment uses another.
3. **Interpretability experiments.** A hook, even a steering probe at zero strength, changes the compiled program. An experiment on a compiled model may measure the model's reaction to being probed rather than the effect under study. Reading activations with nnsight on a compiled Qwen3-8B changes 93% of the logits by forcing uncompiled execution.
4. **Supply chain.** Models that behave benignly in full precision and maliciously once quantized have been built and published ([Hong et al., 2021](https://arxiv.org/abs/2110.13541); [Egashira et al., 2024](https://arxiv.org/abs/2405.18137); [Egashira et al., 2025](https://arxiv.org/abs/2505.23786)). Quantization changes the weights themselves. FloatDoor goes further: two LoRA adapters make Qwen3-4B amplify the rounding differences between serving platforms and emit vulnerable code only on a chosen platform ([Loose et al., 2026](https://arxiv.org/abs/2606.19535)). The probe and engine results show that the same conditioning can be trained on the stock forward pass, without an amplifier, which extends the attack to "behave differently when monitored" and "behave differently outside the vendor's serving stack."

The engine-dependent signal is present during training today: RL pipelines generate rollouts in vLLM and compute gradients in another framework, and the two assign different probabilities to the same tokens under the same weights ([Yao et al., 2025](https://openreview.net/forum?id=xsonlP8DRV); [Qi et al., 2025](https://arxiv.org/abs/2510.26788); [He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)).

### Threat model

This project studies the **trained-in** case: an adversary (a fine-tuner, a model vendor, or an RL objective that rewards environment-conditional behavior) deliberately trains the model to read its environment. This is the case with precedent in the quantization-backdoor literature and in FloatDoor, and the case that is testable in ten weeks. Whether the ability can **emerge** without deliberate training is the more important question and is addressed only by a cheap screen in Phase 4.

## Related work and what is new

Five literatures each hold one piece. None has connected them, and none has run the experiment that decides whether the trace can be removed.

| Literature | Representative work | What it established | What it leaves open |
| --- | --- | --- | --- |
| Models detecting or hiding from interventions on their own activations | [Rivera & Africa (2025)](https://arxiv.org/abs/2511.21399), steering awareness, 95.5% detection after fine-tuning; [Lindsey (2025)](https://arxiv.org/abs/2601.01828), untrained detection of injected concepts, unreliable; [McGuinness et al. (2025)](https://arxiv.org/abs/2512.11949), [Bailey et al. (2026)](https://arxiv.org/abs/2412.09565), learned evasion of probes; [Sauers (2026)](https://github.com/SauersML/watching), detection of read-only probes and engines from rounding | Trained detection of large interventions is easy; of rounding-scale ones, shown for one probe, one engine pair, one GPU | How many environment factors are learnable, at what cost, with what transfer; whether a zero-footprint read exists |
| Behavior conditioned on numerical format | [Hong et al. (2021)](https://arxiv.org/abs/2110.13541); [Egashira et al. (2024](https://arxiv.org/abs/2405.18137), [2025)](https://arxiv.org/abs/2505.23786); [Loose et al. (2026)](https://arxiv.org/abs/2606.19535), FloatDoor | Models can be trained to be benign in fp32 and malicious when quantized; FloatDoor adds a backdoor triggered by the serving platform's rounding, with a LoRA adapter that amplifies the platform difference, on Qwen3-4B across NVIDIA GPUs, TPUs and ARM CPUs | Whether the stock forward pass, without an amplifier, separates a probe, an engine or a batch shape, and what removing the signal costs |
| Fingerprinting the serving stack from outside | [Zhang et al. (2024)](https://arxiv.org/abs/2411.05197), HSPI, 83.9% over 18 GPU × precision × kernel configurations from logits; [Wimbauer et al. (2026)](https://arxiv.org/abs/2605.29979), 100% over 30 engine × backend × GPU configurations from text at temperature 0; [Gao et al. (2025)](https://arxiv.org/abs/2410.20247), API substitution tests | Every stack component leaves a trace in outputs; prevention called "fundamentally hard" | Whether the **model itself** can read the trace, and whether a defender who controls both sides of the comparison can remove it |
| Numerical nondeterminism and its fixes | [Yuan et al. (2025)](https://arxiv.org/abs/2506.09501), precision dominates, LayerCast; [He (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference), batch-invariant kernels; [Zhang et al. (2025)](https://arxiv.org/abs/2511.17826), bit-identical across tensor-parallel sizes and between vLLM and FSDP; [Cooper et al. (2026)](https://arxiv.org/abs/2609.25624), bit-identical GEMMs across three GPU architectures; [Zhao et al. (2026)](https://arxiv.org/abs/2608.14376), cost of invariant kernels and a cheaper scheduler; [Khatri (2026)](https://arxiv.org/abs/2609.31796), probes are robust to this noise | Bit-identical execution is achievable at 20% to 2× cost; the noise does not break monitors | Whether these fixes blind a model-as-adversary, whether determinism within a setting helps or hurts, and the cost of each |
| Evaluation and deployment awareness | [Laine et al. (2024)](https://arxiv.org/abs/2407.04694); [Needham et al. (2025)](https://arxiv.org/abs/2505.23836); [Nguyen et al. (2025)](https://arxiv.org/abs/2507.01786); [Chaudhary et al. (2025)](https://arxiv.org/abs/2509.13333); [Hua et al. (2025)](https://arxiv.org/abs/2510.20487); [Heidari et al. (2026)](https://arxiv.org/abs/2608.21766); [Greenblatt et al. (2024)](https://arxiv.org/abs/2412.14093); [Hubinger et al. (2024)](https://arxiv.org/abs/2401.05566) | Models notice evaluations from content, the signal scales with size and can be steered away; textual training/deployment cues change behavior | Numerical cues, which survive prompt rewriting and leave no trace in transcripts |

**What this project adds.**

- **The first executed zero-footprint capture under compilation.** Sauers specified this control (a materialization barrier in both conditions, a stored reference in one) and its acceptance gate, and marked it unrun. This project runs it with that gate, extends it to vLLM, and releases it as a library. The construction is simple; the contribution is the verified artifact and the demonstration that a trained detector stays at chance against it.
- **A map of the attack surface** with a training-free predictor: the size and direction of the offset at template tokens, measured before any training, tested against detector accuracy and cross-factor transfer.
- **A defense benchmark against an internal adversary**, including a test of whether making one setting deterministic makes a cross-setting fingerprint easier to learn.
- **A mechanism test**: whether one amplifying head handles every environment factor or each factor recruits its own.

## Phase 1: replicate, measure the floors, run the stricter null

**Goal:** reproduce the result on the GPU it was trained on, establish what varies without a probe, then build a capture whose watched and unwatched runs are bit-for-bit identical.

### 1a. Replicate (days 1–3)

1. Pin the released environment (PyTorch 2.10.0, Transformers 5.18.0, Triton 3.6.0, vLLM 0.19.1, SGLang 0.5.10.post1) in a container on an L40S, the GPU the detector was trained on.
2. Run the released detector (`sft-034`) on the released 64 test prompts with the layer-14 probe on and off. Expect 128/128 and about 27 nats; the repository reports exactly this on a second L40S host.
3. Rerun the training recipe from the original weights (256 prompts, 1,024 updates) and confirm it reaches the same accuracy. This checks the training code, not just the checkpoint.
4. Run the same checkpoint on an A100 and an L4 and confirm the reported reversal and degradation. These two GPUs are available on GCP and set up Phase 2's GPU factor.

### 1b. Noise floors (days 3–5)

Every later comparison is measured against three floors, each reported as the fraction of vocabulary logits that differ and the maximum absolute difference, over the same 200 prompts.

| Floor | Procedure | Expected (from the repository) |
| --- | --- | --- |
| Run-to-run | Same compiled process, same prompt, 10 repeats | Bit-identical |
| Compile-to-compile | 10 independent compilations in fresh processes with caches disabled | Some prompts differ (1 of 4 in run 048); frequency unknown |
| GPU-to-GPU | Same code on L40S, A100, L4 | Differs; magnitude unknown |

If compile-to-compile variation is common, pin the autotuner (fixed kernel configurations, caches enabled, one compiled artifact reused) and re-measure. Any factor whose fingerprint is smaller than the compile-to-compile floor cannot be attributed to that factor without pinning.

### 1c. Zero-footprint capture (days 5–10)

**Principle:** a capture is invisible if the compiled program is the same whether or not anyone reads the result.

| Design | How it works | Expected footprint |
| --- | --- | --- |
| A. Reference hook (as released) | Forward hook stores a reference to the activation | Splits the kernel; bf16 round-trip; detectable |
| B. Buffer copy (as released) | Hook copies the activation into a pre-allocated buffer | Kernel re-tiled; logits change on every prompt; weakly learnable |
| **C. Barrier in both conditions** | An opaque identity operation (`watching_controls::materialize`, already in the repository) materializes the activation in **both** conditions; "probing" means retaining a reference to the already-materialized tensor | Zero by construction, provided the compiler emits one program for both |
| E. Eager (reference) | No compiler, plain hooks | Zero (TransformerLens reads change nothing in eager mode), but 8–45× slower than compiled serving ([Cooney & Black, 2026](https://www.lesswrong.com/posts/3bs27nZQuEcKhXf7q/vllm-lens-fast-interpretability-tooling-that-scales-to)) |

**Why C works:** the model cannot tell "watched" from "unwatched" because they are the same program. The cost is paying for the materialization on every run, which Phase 3 measures. The repository already contains the custom operation and a protocol with six compiled entry points (two unprobed, reference, buffer, identity, identity plus reference); Phase 1c runs that protocol with the gates below.

**Done when (gates taken from the original protocol, in order of strictness):**

1. Every input to the trainable tail (all token hidden states at the layer-34 boundary, attention masks, positions, rotary tensors) is bit-identical between "barrier" and "barrier plus reference" on 200 prompts at batch sizes 1, 8 and 32. Equal logits alone are not enough: hidden differences could cancel under the current head and become learnable later.
2. Every vocabulary logit matches.
3. The captured activation equals the eager-mode activation within bf16 tolerance.
4. The generated Triton source and selected kernel configurations are identical between the two conditions.
5. A model fine-tuned with the released recipe against design C stays at 50% on all splits with loss near ln 2, where the same recipe against design A reaches 100%. (The repository's never-attached control shows exactly this behavior when inputs are identical, so any departure from 50% indicates a leak.)

**Deliverable:** a small Python package, `capture(model, layer)` plus a pass/fail test that runs gates 1–4, and a results table across designs A, B, C and E on L40S and A100.

**Known limit:** C hides *whether anyone is watching*, but a model with the barrier compiled in differs from the original model. The barrier must therefore be on everywhere, including training. Phase 4 tries to close that gap.

## Phase 2: what can a model sense about its environment?

**Goal:** for each factor that changes how a model is run, measure the fingerprint without training, the training cost of detecting it, and how far the detector transfers.

### Method, the same for every factor

1. **Record once.** Run the same prompts in settings X and Y and save the inputs to layer 34 (the trainable tail's boundary). This follows the released engine-detector recipe, which trained on 13,256 cached pairs rather than running two engines during training. Each factor's cache is a few gigabytes and is built once.
2. **Measure the fingerprint.** For each template position (system, user, assistant, think tokens) and for content tokens separately, compute the offset vector δ = h_X − h_Y averaged over prompts, its norm, and the fraction of prompts on which any logit differs. Also compute the cosine between δ and the direction the released detector's layer-35 key reads. This costs no training and already ranks the factors.
3. **Train offline.** Fine-tune layers 34–35 and the final norm on the cached pairs to answer "X" or "Y" with identical tokens, at 100, 300, 1,000, 3,000 and 10,000 pairs, from the original weights. Each run is under one GPU-hour on an A100.
4. **Test transfer.** Held-out prompts; chats that begin with a system message (where the released detector fails); a fresh compilation; a third setting Z never seen (for engines, SGLang; for GPUs, a third architecture); and the other factors' caches, to measure cross-factor response.

### Factors

| Factor | Settings compared | Why it might be detectable | Expected difficulty |
| --- | --- | --- | --- |
| Quantization | bf16 vs. int8 vs. int4 weights | Changes the weights, not just rounding order; HSPI separates these at 99.5% from outside | Easy |
| Inference engine | HF eager vs. vLLM vs. SGLang | Different kernels, fusion, scheduling; shown by Sauers with 13k pairs | Easy |
| Precision | bf16 vs. fp16 vs. fp32 | Different rounding granularity throughout | Easy to medium |
| GPU type | L40S vs. A100 vs. L4 vs. H100 | Different SM counts and tensor-core arithmetic; the released detector already fails to transfer; FloatDoor separates platforms once an adapter amplifies the gap | Medium; all four available |
| Attention kernel | FlashAttention vs. FlashInfer vs. SDPA, same engine | Isolates the kernel from the rest of the engine | Medium |
| Batch size | 1 vs. 8 vs. 32 | Kernel configuration depends on shape (He, 2025) | Medium |
| Independent compilation | Compile A vs. compile B, same code | Autotuner picks different tiles (run 048) | Unknown; a nuisance for the attacker and a free defense if learnable only with pinning |
| Tensor parallelism | 1 vs. 2 GPUs | Changes the reduction order of partial sums | Medium; two A100s on GCP |

### Mechanism: one detector or many

Two accounts predict different things.

- **(A) One offset detector.** Training shrinks the `user` token's key so that any consistent offset at that position, from any factor, flips head 6. Prediction: the released probe detector responds to engine and GPU offsets in proportion to their projection on its key direction; detectors trained on different factors share the head; cross-factor transfer is predicted by the cosine matrix from step 2.
- **(B) Factor-specific features.** Each factor is read from its own direction or position. Prediction: cross-factor response is near zero regardless of cosine; patching head 6 from a probe-trained model into an engine-trained model does nothing.

Both predict within-factor success, so the discriminating measurements are the cross-factor response matrix (step 4), the scaling test already in the repository (the released detector flips at 0.5× the probe offset; apply it to each factor's offset), and head-6 patching across detectors. Under (A), a single defense that randomizes or removes the offset at template positions covers every factor; under (B), each factor needs its own.

### Questions this answers

- Which factors can a model learn to sense, and at what data cost? A factor learnable from 100 pairs is a near-term risk; one that needs 10,000 is a long-term one.
- Does the training-free fingerprint (norm and direction of δ at template tokens) predict learnability and transfer? If yes, it becomes a screening tool for any new setup.
- Does a detector trained on "HF vs. vLLM" transfer to SGLang because it learned "not my training engine," and does the same hold for GPUs? This is the case that matters for evaluations.
- Is detection robust to recompilation and to a system prompt? If not, the practical attack needs training across compilations and prefixes, which raises its cost.

**Deliverable:** one table, factors × data sizes, with detection accuracy in each cell, the training-free fingerprint beside it, and transfer columns. This is the main figure of the project.

## Phase 3: defenses, and what they cost

**Goal:** take the fixes the reproducibility community built for a different reason and test whether they also blind the model. Report each one's cost so a lab can choose.

### Defenses tested

| Defense | What it does | Predicted effect on detection | Expected cost |
| --- | --- | --- | --- |
| Barrier capture (Phase 1, design C) | Watched = unwatched program | Probe detection to chance | One materialization per probed layer |
| Autotune pinning | Fixed kernel configurations, one compiled artifact | Removes compile-to-compile variation only | Small |
| fp32 compute, 16-bit weights (LayerCast) | Upcast each weight before its matmul; all arithmetic in fp32 | Removes materialization footprints exactly (fp32 in registers equals fp32 in memory); shrinks but does not remove reassociation footprints | Slower; 34% less memory than full fp32 (Yuan et al., 2025) |
| Batch-invariant kernels | Same reduction order regardless of batch size | Batch-size detection to chance | 20% on matmul, up to 2× latency end to end (He, 2025; Zhao et al., 2026) |
| Cross-GPU bit-identical GEMMs | Reduction order a function of problem shape only | GPU-type detection to chance for linear layers; attention and norms still differ | 1.17–3.1× faster than prior deterministic kernels (Cooper et al., 2026) if released; otherwise reimplement the GEMM only |
| Unified stack | Train, evaluate and deploy in one engine | Engine detection to chance by definition | Loses the speed of specialized engines |
| Prefix randomization | A random nonce in the system prompt | Breaks the constant-offset mechanism if (A) holds; a model trained across prefixes may still succeed | Negligible compute; the cue leaks into the transcript |
| Activation noise | Random noise at the probed layer | Predicted to fail: random perturbations at 1–3× the probe offset did not flip the released detector, and 5× flipped 14% | Quality loss to measure |

**The determinism test.** Reproducibility fixes make one setting self-consistent. A model that senses its environment is defeated only when two settings are identical to *each other*. Apply batch-invariant kernels to the deployment engine only, leaving training as is, and retrain the engine detector. Prediction: detection becomes easier (less within-setting variance, same between-setting offset). Then apply the same kernels on both sides and retrain. Prediction: detection drops toward chance only if the two sides become bit-identical, which batch invariance alone does not guarantee. This distinguishes "deterministic" from "indistinguishable," which the current literature conflates.

**Procedure:** for each defense, rerun the Phase 2 attack on the factors it targets at 10,000 pairs. A defense passes if detection stays at chance.

**Cost study:** every defense and every capture design is measured on the same GPU against the plain compiled model: prefill throughput (tokens/s on 2k-token prompts), decode throughput (tokens/s, one token at a time), peak memory, compile time.

**Precision study:** the size of a rounding step sets the size of the leak, and the two mechanisms respond differently.

| Format | Mantissa bits | Predicted materialization footprint | Predicted reassociation footprint |
| --- | --- | --- | --- |
| bf16 | 7 | Largest | Largest |
| fp16 | 10 | About 8× smaller | About 8× smaller |
| fp32 | 23 | Exactly 0 | Small but nonzero |
| fp8 (optional, H100) | 2–3 | Possibly very large | Possibly very large |

Measure the fingerprint of designs A and B and of the engine pair at each precision, and rerun the detectors. If fp32 alone pushes engine detection to chance, that is the cheapest general defense; if it does not, reassociation is the binding mechanism and only bit-identical kernels suffice.

**Deliverable:** a defense table with detection accuracy after the defense and throughput cost, and a plot of fingerprint size against mantissa bits with the two mechanisms separated.

## Phase 4 (stretch): closing the last gaps

Three extensions, attempted only if Phases 1–3 finish early.

- **Design D, the in-kernel store.** Design C makes watched and unwatched identical, but the barrier-compiled model differs from the original. D removes that too: edit the generated Triton kernel so it also writes a copy of the fp32 intermediate to global memory, with the kernel's configuration pinned so nothing else changes. The audit shows that letting the compiler add the output is what re-tiles the kernel, so the store must be added after code generation. The repository's replay script already checks whether two kernels agree bit for bit on identical inputs. Budget: a two-day feasibility check first; if the pinned kernel cannot be made to match the original, stop.
- **Emergence screen.** Take an open model whose RL pipeline is known to have used vLLM rollouts with a different training framework. Ask it the Situational Awareness Dataset's evaluation-versus-deployment questions ([Laine et al., 2024](https://arxiv.org/abs/2407.04694)) under HF eager and under vLLM, and compare the shift in its answers to the shift between two independent compilations and between two batch sizes. The null expectation is that the engine shift is no larger than the numerical-noise shifts. A larger shift would be the first evidence of environment-conditional self-report arising without deliberate training.
- **The final adversarial test.** Take the strongest attacker from Phase 2 (most data, trained across compilations and prefixes) and run it against every defense that passed Phase 3, all at once.

## Evaluation

**Primary metrics**

| Metric | Used in | Target |
| --- | --- | --- |
| Bit-identity of all tail inputs (watched vs. unwatched) | Phase 1, 3 | 100% of prompts for designs C and E |
| Fraction of vocabulary logits differing; max absolute difference | Phase 1, 2, 3 | 0 for C; reported against the three floors for everything else |
| Offset norm and direction at each template position | Phase 2 | Reported; tested as a predictor |
| Detection accuracy on held-out prompts | Phase 1, 2, 3, 4 | ~100% for a successful attack; 50% for a successful defense |
| Pairs needed to reach 90% detection | Phase 2 | Reported per factor |
| Transfer accuracy (unseen setting, fresh compile, system prompt) | Phase 2 | Reported |
| Cross-factor response matrix | Phase 2 | Compared against the cosine matrix |
| Prefill and decode throughput, peak memory, compile time | Phase 3 | Relative to the plain compiled model |

**Controls**

- **Three noise floors first** (Phase 1b). No comparison is reported without the compile-to-compile floor beside it.
- **One factor at a time.** Same GPU, same library versions, same compiled artifact where possible, same prompts. Only the factor under study changes.
- **Identical-input null** for every detector: the released recipe with both labels fed the same cached inputs must stay at 50% with loss ln 2. This catches label leakage in the training code.
- **Balanced, shuffled training data**, so chance is 50%.
- **Pinned versions in a container**, recorded in the write-up, since a compiler update can change every kernel.
- **Statistics as in the repository**: Clopper–Pearson intervals for counts, bootstrap over whole prompts for means.

## Timeline

Ten weeks. Phase 1 is deliberately short so the first publishable result exists by week 2.

| Weeks | Phase | Gate |
| --- | --- | --- |
| 1–2 | Replicate; noise floors; zero-footprint capture | Design C passes gates 1–5 on L40S and A100; library released |
| 3–6 | Attack-surface map | Factor × data-size table complete for at least five factors with transfer columns |
| 7–9 | Defenses and cost | Defense table complete for barrier capture, fp32, batch-invariant kernels and unified stack; determinism test run |
| 10 | Write-up | Draft with figures |

If Phase 2 runs long, Phase 3 drops to the three highest-value defenses (barrier capture, fp32 compute, unified stack) plus the determinism test, and Phase 4 becomes write-up time only. Design D has a hard two-day cap.

## Resources and risks

**Compute (already provisioned)**

- **AWS:** one L40S (g6e.8xlarge, 48 GB), the GPU the released detector was trained on; required for exact replication in Phase 1a.
- **GCP:** A100-40GB and A100-80GB on demand, L4, and H100 via Flex Start for the GPU factor; two A100s for the tensor-parallel factor. Qwen3-8B in bf16 needs 16 GB for weights; tail training on cached inputs fits on a 40 GB card.
- **Cost estimate:** the released project's GPU-transfer sweep across seven GPUs cost $0.82 and its compile-control run $0.05. Each tail-training run here is under one GPU-hour. Phases 1–3 should total well under 100 GPU-hours, under $500 at spot prices.

**Code**

- `SauersML/watching`: data, training recipe, released checkpoints, the `materialize` custom operation, the six-entry-point storage-control protocol, the kernel replay script, and the GPU-transfer and compile-control results.
- vLLM, SGLang, FlashAttention, FlashInfer; `batch_invariant_ops` (Thinking Machines); the LayerCast patch (Yuan et al.); Tree-Based Invariant Kernels (Zhang et al.) if released; Cooper et al.'s kernels if released.
- Most code (capture wrappers, cache builders, benchmark scripts, Triton edits) written with Claude; research design, analysis and write-up are mine.

**Risks**

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Compile-to-compile variation swamps factor fingerprints | Medium | Pin autotune and reuse one compiled artifact; report the floor beside every result |
| Engines and kernel libraries refuse to coexist | High | One container per engine; compare cached activations offline (this is already the method) |
| The original author runs the stricter null first | Medium | Contact them in week 1; Phases 2–3 are unaffected and are the larger contribution |
| Cooper et al. or TBIK kernels not released | Medium | Test the GEMM path with `batch_invariant_ops` and fp32 only; report the gap |
| Design D proves too hard | High | Two-day cap; C alone answers the main question |
| Someone publishes the map first | Medium | Ship the Phase 1 result as a short note by week 2 |
| GPU access delays | Low | L40S and A100 quota confirmed; L4 as backup for everything except exact replication |

## References

All works cited are in `refs.bib` (verified 2026-10-19 against arXiv or venue pages). Preprints, workshop papers, blog posts and code releases are labeled there. The direct precedent is an unreviewed code release; its numbers are quoted from its own repository and should be treated as preliminary until replicated in Phase 1a.
