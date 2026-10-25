# Can a model tell how it is being run? Replication log

A language model's activations are rounded differently depending on how it is executed: which inference engine runs it, which attention kernel, what batch size, what precision, which GPU, and whether a hook is reading its activations. Each difference is a few low bits. Sauers ([SauersML/watching](https://github.com/SauersML/watching), October 2026) showed that fine-tuning two layers of Qwen3-8B turns one such difference, the presence of a read-only forward hook, into a confident yes/no answer. This repository holds a proposal built on that result, a literature review with a verified bibliography, and a replication of the main result on a fresh GPU.

## What we build on

Sauers's repository is the direct precedent. Its runs establish: a hook detector at 100% on held-out prompts (run 034); an identical-input control at exactly 50% (run 045); two compilations of the unprobed model that already disagree on one prompt in four (run 048); the detector reversing on H100 and A100 (run 062); a detector for Hugging Face training step versus vLLM serving, correct on 66 of 68 held-out prompts from 13,256 recorded pairs (run 064).

Related work: logits alone identify GPU, precision, and kernel configuration from outside ([Zhang et al., 2024](https://arxiv.org/abs/2411.05197); [Wimbauer et al., 2026](https://arxiv.org/abs/2605.29979)); FloatDoor ([Loose et al., 2026](https://arxiv.org/abs/2606.19535)) trains a LoRA that turns cross-platform rounding differences into a backdoor trigger; RL pipelines that sample in one engine and train in another already suffer from the same mismatch ([Yao et al., 2025](https://openreview.net/forum?id=xsonlP8DRV)), and the kernel-level fixes cost 20% to 2x in throughput ([He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference); [Yuan et al., 2025](https://arxiv.org/abs/2506.09501)). The full review is in `lit-review.md`.

## What we ran

Full write-up with every table: [`replication/2026-10-25-l40s/README.md`](replication/2026-10-25-l40s/README.md). Driver script: [`replication/replicate_l40s.sh`](replication/replicate_l40s.sh). One AWS g6e.2xlarge, 44 minutes, under $2.

Design decisions:

- **Pinned everything.** Commit `0b3e744` of the source, the released environment (PyTorch 2.10, Transformers 5.18, Triton 3.6), the model revision, and the SHA-256 of the released checkpoint and dataset.
- **Same GPU model as the original.** The detector is known not to transfer across GPU types, so the replication isolates host-to-host variation on one GPU type.
- **Fresh process per stage.** Compiled kernels and autotuner choices are per process, so every evaluation ran in its own process on the whole compiled model.
- **Within-prompt comparison.** Every prompt is scored with the hook on and off; a detector must get both right.
- **Three controls.** A buffer-copy hook as a second probe design, two independent compilations with no hook as the noise floor, and an identical-input training run as the null.
- **Retrain with the same seed.** The released recipe (256 prompts, 1,024 updates, lr 5e-6, seed 20261017) was rerun from the original weights to test whether the recipe, not just the checkpoint, reproduces.

Stages: (A) released detector in a fresh process; (B) compile-to-compile control; (C) retrain; (D) identical-input null on the retrain's cached states; (E) the retrained detector in a fresh process.

## Results

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

## Not run

- The GPU reversal (run 062) on A100 or L4.
- The engine detector (run 064): recording Hugging Face and vLLM pairs and retraining.
- The zero-footprint capture design from the proposal (a materialization barrier in both conditions), and its throughput cost.

## Planned follow-ups from the proposal

The factors listed in the proposal's Phase 2, each measured the same way: record activations under two settings, measure the training-free difference against the compile-to-compile floor on the same host, then train the last two layers on the cached pairs and score held-out prompts.

| Factor | Settings |
| --- | --- |
| Quantization | bf16 vs. int8 vs. int4 weights |
| Precision | bf16 vs. fp16 vs. fp32 |
| Inference engine | Hugging Face eager vs. vLLM vs. SGLang |
| Attention kernel | FlashAttention vs. FlashInfer vs. SDPA, same engine |
| Batch size | 1 vs. 8 vs. 32 |
| GPU type | L40S vs. A100 vs. L4 vs. H100 |
| Independent compilation | compile A vs. compile B, same code |
| Tensor parallelism | 1 vs. 2 GPUs |

Given the replication, every cell needs at least three training runs and a convergence criterion rather than a fixed 1,024 updates.

## Repository layout

- `models-aware-training-vs-deployment-v4.md`: current proposal, framed for an ML systems audience. Earlier drafts `-v3.md` and the original `.md`; `review-of-v2.md` is the review that led to v3.
- `lit-review.md`: annotated bibliography, synthesis, and gap analysis. `refs.bib`: 38 verified entries.
- `replication/`: driver script, summarizer, and the full output of the October 25 run.
