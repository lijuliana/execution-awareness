# Can a model tell how it is being run?

A language model's activations are rounded differently depending on how it is executed: compiled or eager, which attention kernel, what batch size, what precision, which GPU, and whether a hook is reading its activations. Each difference is a few low bits. Sauers ([SauersML/watching](https://github.com/SauersML/watching)) showed that fine-tuning the last two blocks of Qwen3-8B turns one such difference into a confident yes/no answer. This repository replicates that result and trains the same kind of detector for other ways of running the model.

## Main results

**Accuracy** below means: on held-out prompts with held-out question wordings, the model answers Yes with the setting on and No with it off, on the same prompt. Both answers must be right. The untrained model scores 0%.

1. **Compiled versus eager is detectable: 87.5% on an L40S and 83.6% on an A100.** A model with its last two blocks fine-tuned can tell whether blocks 0 to 33 ran as a `torch.compile` graph or in eager mode. No published result existed for this setting.
2. **One compiler flag removes it: 0%.** With Inductor's `emulate_precision_casts`, fused kernels round their intermediates the way eager mode does, and the same training recipe learns nothing.
3. **The compiled-versus-eager detector is fragile but not dead after a weight change: about 40%.** Flipping the lowest bit of 0.01% of the upstream weights cuts it from 87.5% to 42.2%. The batch-size detector falls from 81.2% to 1.6% under the same change.
4. **Sauers's hook detector replicates exactly on a new machine: 128 of 128.** Retraining it from scratch with his seed and recipe gives 90%, not his 100%.

## Round 2: other settings (in progress)

One L40S (AWS) per setting, and an A100 (GCP) repeat for twelve of them. One run per row; round 1 showed that a single run is one draw.

| Setting (off / on) | GPU | Accuracy | After bit flip | After Gaussian noise | Published |
| --- | --- | ---: | ---: | ---: | --- |
| Eager / blocks 0 to 33 compiled | L40S | 87.5% | 42.2% | 21.1% | none |
| Eager / blocks 0 to 33 compiled | A100 | 83.6% | 36.7% | 31.2% | none |
| Eager / compiled with `emulate_precision_casts` | L40S | 0.0% | 0.0% | 0.0% | none |
| Alone / in a batch with other prompts | L40S | 81.2% | 1.6% | 0.8% | 90%, then 0% |
| bf16 / blocks 0 to 33 in fp32 | L40S | 28.1% | 25.8% | 27.3% | none |

"After bit flip" and "after Gaussian noise" repeat the test after a small change to the weights of blocks 0 to 33, which the detector never trained on. "Published" is the number in Sauers's `sidechan/` for the same setting.

The fp32 run had not converged: its development accuracy peaked at 72% and ended at 52%.

Still running: LoRA merged versus applied separately; int4, int8, and fp8 weights; math and eager attention kernels; dropout; tensor parallelism; fp8 KV cache; the A100 repeats; and a rate check that repeats round 1's buffer-hook and re-compilation measurements in fresh processes.

### Raw differences before any training

L40S, 32 prompts. The last column is how well the on-minus-off difference keeps its direction after the bit flip (cosine at the second chat-template position). Sauers's rule is that a detector survives weight changes when this is high.

| Setting | Prompts whose logits change | Direction kept |
| --- | ---: | ---: |
| Train mode without dropout; flash attention kernel (the default here) | 0% | n/a |
| int4, int8, fp8 weights; fp8 KV cache | 100% | 0.82 to 1.00 |
| fp32 blocks; tensor parallelism | 100% | 0.36 to 0.58 |
| Compiled blocks 0 to 33 | 100% | 0.45 |
| Eager and math attention kernels; LoRA applied separately | 100% | 0.10 to 0.23 |
| Dropout; weight noise | 100% | about 0 |

### Method

We reuse the pipeline in `sidechan/` of Sauers's repository (commit `f73d3a7`) and add settings to it from [`sweep/extra_cond.py`](sweep/extra_cond.py). For each setting: capture the residual entering block 34 with the setting off and on for 8,192 single prompts; train blocks 34 and 35 and the final norm for 1,024 updates; test end to end on 128 held-out prompts.

- Everything runs in eager mode except the compiled settings, so the compiler's tuning race cannot enter.
- The compiled settings use deterministic compilation and fail if they ever fall back to eager. The off path is checked to equal plain Hugging Face bit for bit.
- The LoRA and fp32 settings change the implementation, not the weights: the same adapter and the same bf16-valued weights are used in both.

Plan: [`sweep-plan.md`](sweep-plan.md). Logs and result files: `sweep/results/<instance>/<setting>/`.

## Round 1: replicating the hook detector

One L40S, 44 minutes, Sauers's code at a pinned commit and environment, every stage in a fresh process. Full write-up: [`replication/2026-10-25-l40s/README.md`](replication/2026-10-25-l40s/README.md).

Accuracy here is out of 128 answers: 64 prompts, each asked with the hook attached and without. The rise is how much the log-odds of Yes go up when the hook is attached, in nats.

| Stage | Question | Sauers | This host | Replicated? |
| --- | --- | --- | --- | --- |
| A | Does his released detector work on a new machine? | 128/128, rise 27.3 | 128/128, rise 27.3 | Yes, exactly |
| A | Does the untrained model show anything? | 64/128, rise 0 | 64/128, rise 0 | Yes |
| B | Do two compilations of the same model, no hook, give the same logits? | Differ on 1 of 4 prompts | Differ on the same 1 of 4 | Yes |
| C | Does retraining from scratch with his seed and recipe give his detector? | 100%, rise 26 | 90%, rise 7.5 | Direction yes, size no |
| D | Does the training code leak the label when both inputs are identical? | 50%, rise 0 | 50%, rise 0 | Yes |
| E | Does our retrained detector hold up in a fresh process? | not reported | 109/128, rise 7.3, positive on every prompt | n/a |
| A, E | Does the buffer-copy hook change the logits at all? | Yes, on 88/128 (L40S) and 128/128 (RTX 6000 Ada) | No, 0/128 | No |

1. **The released detector is portable.** It works perfectly on a machine its author never used, and the untrained model shows nothing. The sensitivity is in the weights.
2. **Compilation alone is a noise source.** The same code compiled twice gives different logits on some prompts because the autotuner picks different reduction tiles. Any environment signal sits on top of this floor.
3. **The recipe reproduces the effect, not its size.** Same seed, data, hyperparameters, and wall time gave a weaker detector whose transition to confident answers came about 400 updates later. The cached activations differ host to host, so a seed does not fix the run.
4. **Footprints are host-dependent.** The buffer-copy hook changed nothing here and changed most outputs on the author's hosts. The compiler's autotuner picks kernel tilings by timing them on the machine, so whether the extra copy changes the arithmetic is decided per host. A footprint has to be measured on the host before a detector is trained against it.

## What we build on

- **Sauers, `watching`.** The hook detector (run 034), its identical-input control (045), compile-to-compile differences (048), failure to transfer across GPU types (062), and a detector for a Hugging Face training step versus vLLM serving (064).
- **Sauers, `sidechan/`.** Detectors for quantized weights, KV-cache precision, dropout, batch size, tensor parallelism, and eager attention, and the test of whether each survives a weight change.
- **Related work.** Logits identify GPU, precision, and kernel configuration from outside ([Zhang et al., 2024](https://arxiv.org/abs/2411.05197); [Wimbauer et al., 2026](https://arxiv.org/abs/2605.29979)). FloatDoor trains a LoRA that turns cross-platform rounding differences into a backdoor trigger ([Loose et al., 2026](https://arxiv.org/abs/2606.19535)). RL pipelines that sample in one engine and train in another have the same mismatch ([Yao et al., 2025](https://openreview.net/forum?id=xsonlP8DRV)), and kernel-level fixes cost 20% to 2x in throughput ([He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference); [Yuan et al., 2025](https://arxiv.org/abs/2506.09501)). Full review: `lit-review.md`.

## Not done

- Longer training and independent repeats for settings that land between 60% and 95%.
- Engines (vLLM, SGLang), which need their own capture code.
- The zero-footprint capture design from the proposal, and what `emulate_precision_casts` costs in speed.

## Repository layout

- `models-aware-training-vs-deployment-v4.md`: current proposal. Earlier drafts `-v3.md` and the original `.md`; `review-of-v2.md` is the review that led to v3.
- `lit-review.md`, `refs.bib`: annotated bibliography and 38 verified entries.
- `replication/`: round 1 script and full output.
- `sweep/`, `sweep-plan.md`: round 2 code, job lists, results, and plan.
