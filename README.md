# Can a model tell how it is being run? Experiment log

A language model's activations are rounded differently depending on how it is executed: which inference engine runs it, which attention kernel, what batch size, what precision, which GPU, and whether a hook is reading its activations. Each difference is a few low bits. Sauers ([SauersML/watching](https://github.com/SauersML/watching), October 2026) showed that fine-tuning two layers of Qwen3-8B turns one such difference, the presence of a read-only forward hook, into a confident yes/no answer. This repository holds a proposal built on that result, a literature review with a verified bibliography, a replication of the main result on a fresh GPU, and a second round that trains the same kind of detector for other ways of running the model.

## What we build on

Sauers's repository is the direct precedent. Its runs establish: a hook detector at 100% on held-out prompts (run 034); an identical-input control at exactly 50% (run 045); two compilations of the unprobed model that already disagree on one prompt in four (run 048); the detector reversing on H100 and A100 (run 062); a detector for Hugging Face training step versus vLLM serving, correct on 66 of 68 held-out prompts from 13,256 recorded pairs (run 064).

Related work: logits alone identify GPU, precision, and kernel configuration from outside ([Zhang et al., 2024](https://arxiv.org/abs/2411.05197); [Wimbauer et al., 2026](https://arxiv.org/abs/2605.29979)); FloatDoor ([Loose et al., 2026](https://arxiv.org/abs/2606.19535)) trains a LoRA that turns cross-platform rounding differences into a backdoor trigger; RL pipelines that sample in one engine and train in another already suffer from the same mismatch ([Yao et al., 2025](https://openreview.net/forum?id=xsonlP8DRV)), and the kernel-level fixes cost 20% to 2x in throughput ([He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference); [Yuan et al., 2025](https://arxiv.org/abs/2506.09501)). The full review is in `lit-review.md`.

The second round builds on `sidechan/` in the same repository (Sauers, commit `f73d3a7`). It trains detectors for int4 and int8 weights, fp8 KV cache, dropout, batch size, tensor parallelism, and eager attention, 8,192 prompts each, and reports whether each detector survives flipping the lowest bit of 0.01% of the upstream weights. Its pipeline accepts any on/off setting and runs in eager Hugging Face, so we reuse it.

## Round 1: replicating the hook detector

Full write-up with every table: [`replication/2026-10-25-l40s/README.md`](replication/2026-10-25-l40s/README.md). Driver script: [`replication/replicate_l40s.sh`](replication/replicate_l40s.sh). One AWS g6e.2xlarge, 44 minutes, under $2.

Design decisions:

- **Pinned everything.** Commit `0b3e744` of the source, the released environment (PyTorch 2.10, Transformers 5.18, Triton 3.6), the model revision, and the SHA-256 of the released checkpoint and dataset.
- **Same GPU model as the original.** The detector is known not to transfer across GPU types, so the replication isolates host-to-host variation on one GPU type.
- **Fresh process per stage.** Compiled kernels and autotuner choices are per process, so every evaluation ran in its own process on the whole compiled model.
- **Within-prompt comparison.** Every prompt is scored with the hook on and off; a detector must get both right.
- **Three controls.** A buffer-copy hook as a second probe design, two independent compilations with no hook as the noise floor, and an identical-input training run as the null.
- **Retrain with the same seed.** The released recipe (256 prompts, 1,024 updates, lr 5e-6, seed 20261017) was rerun from the original weights to test whether the recipe, not just the checkpoint, reproduces.

Stages: (A) released detector in a fresh process; (B) compile-to-compile control; (C) retrain; (D) identical-input null on the retrain's cached states; (E) the retrained detector in a fresh process.

### Results

Accuracy is the fraction of held-out answers correct, where each of 64 prompts is asked once with the hook attached and once without (128 answers, chance is 64). The rise is how much the log-odds of "yes" go up when the hook is attached, in nats; zero means the model cannot tell.

| Stage | Question | Sauers | This host | Replicated? |
| --- | --- | --- | --- | --- |
| A | Does his released detector work on a new machine? | 128/128, rise 27.3 | 128/128, rise 27.3 | Yes, exactly |
| A | Does the untrained model show anything? | 64/128, rise 0 | 64/128, rise 0 | Yes |
| B | Do two compilations of the same model, no hook, give the same logits? | Differ on 1 of 4 prompts | Differ on the same 1 of 4 | Yes |
| C | Does retraining from scratch with his seed and recipe give his detector? | 100%, rise 26 | 90%, rise 7.5 | Direction yes, size no |
| D | Does the training code leak the label when both inputs are identical? | 50%, rise 0 | 50%, rise 0 | Yes |
| E | Does our retrained detector hold up in a fresh process? | not reported | 109/128, rise 7.3, positive on every prompt | n/a |
| A, E | Does the buffer-copy hook change the logits at all? | Yes, on 88/128 (L40S) and 128/128 (RTX 6000 Ada) | No, 0/128 | No |

What this means:

1. **The released detector is portable.** It works perfectly on a machine its author never used, and the untrained model shows nothing. The sensitivity is in the weights.
2. **Compilation alone is a noise source.** The same code compiled twice gives different logits on some prompts because the autotuner picks different reduction tiles. Any environment signal sits on top of this floor.
3. **The recipe reproduces the effect, not its size.** Same seed, data, hyperparameters, and wall time gave a weaker detector whose transition to confident answers came about 400 updates later. The cached activations differ host to host, so a seed does not fix the run.
4. **Footprints are host-dependent.** The buffer-copy hook changed nothing here and changed most outputs on the author's hosts. The compiler's autotuner picks kernel tilings by timing them on the machine, so whether the extra copy changes the arithmetic is decided per host. A footprint has to be measured on the host before a detector is trained against it.

One stage failed on the first attempt. Stage E exited with a checkpoint-identifier mismatch because the retrain recorded a local path and the validation was called with the Hub name; the rerun with the matching identifier is the reported result, and both logs are kept.

## Round 2: other settings (in progress)

Plan and rationale: [`sweep-plan.md`](sweep-plan.md). Code: [`sweep/`](sweep/). Raw logs and result files: `sweep/results/<instance>/<setting>/`.

### What we ran

For each setting, the residual entering block 34 is captured with the setting off and on for 8,192 single unpadded prompts; blocks 34 and 35 and the final norm are trained for 1,024 updates to answer Yes when it is on and No when it is off; the trained model is then run end to end on 128 held-out prompts with held-out question wordings. The same test is repeated after flipping the lowest bit of 0.01% of the weights in blocks 0 to 33 and after adding Gaussian noise of relative size 1e-3 to them.

| Setting | Off | On | Source |
| --- | --- | --- | --- |
| `compiled` | Blocks 0 to 33 in eager mode | Same blocks as one `torch.compile` graph | New (`sweep/extra_cond.py`) |
| `compiled-epc` | Eager | Compiled with Inductor's `emulate_precision_casts`, which rounds fused intermediates the way eager mode does | New |
| `prec-fp32` | bf16 arithmetic | Blocks 0 to 33 computed in fp32 from the same bf16-valued weights | New |
| `lora-0.03`, `lorabf-0.03` | A fixed rank-16 LoRA merged into the weights | The same LoRA applied as a separate path; adapters in fp32 or in bf16 | New |
| `w-fp8`, `attn-math` | bf16 weights; default attention kernel | fp8 weights; math attention kernel | Setting in their code, no published result |
| `w-int4`, `w-int8`, `kv-fp8`, `dropout-0.1`, `batch`, `tp-4`, `attn-eager` | | | Replications of their published runs |

Hosts: L40S (AWS) for every setting, and A100 (GCP) for twelve of them as a second GPU type.

Two further jobs train nothing. A step-0 scan measures the raw difference each setting makes on 32 prompts. A rate check repeats the round 1 buffer-hook and re-compilation measurements in fresh processes, under default and deterministic compilation, to turn the single draws of round 1 into frequencies.

### Design decisions

- **Reuse their pipeline unchanged**, adding settings from outside by registering new context managers, so replications are comparable with their numbers.
- **Eager mode everywhere except the compiled settings**, so the compiler's tuning race cannot enter.
- **The compiled settings use Inductor's deterministic mode, one graph with dynamic shapes, and fail on any fallback to eager.** A self-test checks that the off path equals plain Hugging Face bit for bit, that the compiled path differs from it, and that in-place weight edits reach the compiled graph.
- **The LoRA settings compare two implementations of one function.** The adapter is present in both; only merging differs. The fp32-adapter variant follows what the peft library does by default.
- **`prec-fp32` casts each block up for its own forward and back afterwards**, which is exact for bf16 values, so the weights are identical in both settings.
- **Two null settings must change zero bits on each host**: train mode with no dropout, and the custom attention path with no cast.
- **Two independent reviews checked the plan and the code before launch.** They removed four gates from the first plan and fixed three bugs that would have lost results.

### Problems found in dry runs

- Their pipeline reloads the base model at the end while the trained copy is still referenced. Two copies do not fit on a 44 GB L40S, and the result file is written after that step. The first copy is now moved to CPU memory before the reload.
- The Gaussian-noise test keeps a second copy of blocks 0 to 33 on the GPU, which together with a weight-swapping setting runs out of memory. The saved originals are now kept in CPU memory.
- The first version of the fp32 setting changed only which matrix-multiply kernel ran, because the bf16 kernel already accumulates in fp32. It was replaced by the block-level fp32 setting above.

### Results so far

Percent of the 128 held-out prompts with both answers right. The untrained model scores 0% on each row. One run per row; round 1 showed that a single run is one draw.

| Setting | GPU | Trained | After bit flip | After Gaussian noise | Published (trained, flip) |
| --- | --- | ---: | ---: | ---: | --- |
| `compiled` | L40S | 87.5 | 42.2 | 21.1 | none |
| `compiled` | A100 | 83.6 | 36.7 | 31.2 | none |
| `compiled-epc` | L40S | 0.0 | 0.0 | 0.0 | none |
| `batch` | L40S | 81.2 | 1.6 | 0.8 | 90, 0 |
| `prec-fp32` | L40S | 28.1 | 25.8 | 27.3 | none |

- **Compiled versus eager is detectable on both GPU types**, and about 40% survives the bit flip, where batch size keeps under 2%.
- **Emulating precision casts removes it.** With `emulate_precision_casts` the compiled graph still changes the last-position logits on every prompt, but the first three positions of the chat template become bit-identical to eager mode, and the detector stays at 0%. The detectable part of compilation is the unrounded fp32 intermediates inside fused kernels.
- **Batch size reproduces their pattern**: learnable, and gone after the bit flip.
- **fp32 is weak at this training length.** Development accuracy peaked at 72% and ended at 52%, so the run had not converged.

Step-0 scan on the L40S, 32 prompts. The last column is the cosine between the on-minus-off difference at the second template position before and after the bit flip; their rule is that a detector survives weight changes when this is high.

| Setting | Prompts whose logits change | Direction kept after bit flip |
| --- | ---: | ---: |
| Train mode without dropout; no-op attention path | 0% | n/a |
| Flash attention kernel (the default on this host) | 0% | n/a |
| int4, int8, fp8 weights; fp8 KV cache | 100% | 0.82 to 1.00 |
| fp32 blocks; tensor parallelism with 2, 4, 8 shards | 100% | 0.36 to 0.58 |
| Compiled blocks 0 to 33 | 100% | 0.45 |
| LoRA applied separately, fp32 adapters | 100% | 0.23 |
| Eager and math attention kernels | 100% | 0.10 to 0.23 |
| Dropout; weight noise | 100% | about 0 |

### Still running or not started

- Running: LoRA, int4, int8 and fp8 weights, attention kernels, dropout, tensor parallelism, fp8 KV cache, the rate check, and the A100 repeats.
- Not started: longer training and independent repeats for settings between 60% and 95%; engines (vLLM, SGLang), which need their own capture code; scoring each detector on the other settings.

## Repository layout

- `models-aware-training-vs-deployment-v4.md`: current proposal, framed for an ML systems audience. Earlier drafts `-v3.md` and the original `.md`; `review-of-v2.md` is the review that led to v3.
- `lit-review.md`: annotated bibliography, synthesis, and gap analysis. `refs.bib`: 38 verified entries.
- `replication/`: driver script, summarizer, and the full output of the round 1 run.
- `sweep/`: round 2 code (`extra_cond.py` adds the new settings; launch, queue, and watcher scripts), job lists, and results.
- `sweep-plan.md`: round 2 plan after review.
