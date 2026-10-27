# Can a model tell how it is being run?

The same model gives slightly different numbers depending on how it is executed: compiled or eager, which attention kernel, what batch size, what precision, whether its weights are quantized. Each difference is a few low bits in the activations. Sauers ([SauersML/watching](https://github.com/SauersML/watching)) showed that fine-tuning the last two blocks of Qwen3-8B turns one such difference into a confident yes/no answer. This repository replicates that result (round 1) and trains the same kind of detector for eight other kinds of execution setting, in fourteen variants (round 2).

## Short answer

- **Every kind of setting we tried can be detected in at least one form.** Eight kinds, fourteen variants. After fine-tuning its last two blocks, the model answers correctly whether the setting is on, with single-run accuracy from 28% to 100%; the untrained model is at or near 0%. The one variant it cannot detect is compilation with precision casts emulated.
- **Compiled versus eager: 87.5% on an L40S, 83.6% on an A100.** The model can tell whether its first 34 blocks ran through `torch.compile`.
- **One compiler flag makes that undetectable: 0%.** `emulate_precision_casts` makes fused kernels round the way eager mode does.
- **Whether a LoRA is merged into the weights or applied separately: 63% to 91%.**
- **Quantized weights and dropout stay at 98% to 100% when the weights change. The settings that only reorder arithmetic fall to under 20%**, matching Sauers's pattern. Compiled-versus-eager and fp32 keep about a third.

## Round 2 results, by setting (in progress)

**How to read this.** For each setting the model is asked a yes/no question about how it is being run. Each of 128 held-out prompts is run twice, once with the setting on and once with it off. A prompt counts only if the model says Yes when it is on and No when it is off. The table gives the percent of prompts that count.

"After a weight change" repeats the test after flipping the lowest bit of 0.01% of the weights in the blocks the detector does not own. A detector that survives this could keep working while a model is being trained. A detector that does not is tied to one exact set of weights.

Each cell is one training run. "Theirs" is the number Sauers published for the same setting; "new" means no published number existed.

### Detected, and still detected after the weights change

| The model is asked whether... | L40S | A100 | After a weight change (L40S / A100) | Prior result |
| --- | ---: | ---: | ---: | --- |
| its weights are stored in fp8 | 99% | 100% | 99% / 100% | new |
| its weights are quantized to int4 | running | 100% | running / 100% | theirs: 100% |
| dropout is on | 98% | running | 99% / running | theirs: 100% |
| its weights are quantized to int8 | running | running | | theirs: 99% |
| its KV cache is stored in fp8 | queued | not run | | theirs: 100% |

### Detected, but mostly lost when the weights change

| The model is asked whether... | L40S | A100 | After a weight change (L40S / A100) | Prior result |
| --- | ---: | ---: | ---: | --- |
| each layer is split into 4 shards (tensor parallelism) | 88% | 98% | 17% / 1% | theirs: 59% |
| its LoRA is applied separately, not merged (bf16 adapter) | 91% | not run | 6% | new |
| attention uses the math kernel, not the default | 91% | running | 0% | new |
| the prompt shares a batch with other prompts | 81% | 53% | 2% / 16% | theirs: 90% |
| its LoRA is applied separately, not merged (fp32 adapter) | 63% | 79% | 2% / 4% | new |
| attention uses the eager code path, not the default | queued | 61% | queued / 6% | theirs: 59% |

### In between: partly survives a weight change

| The model is asked whether... | L40S | A100 | After a weight change (L40S / A100) | Prior result |
| --- | ---: | ---: | ---: | --- |
| blocks 0 to 33 were compiled with `torch.compile` | 88% | 84% | 42% / 37% | new |
| blocks 0 to 33 were computed in fp32, not bf16 | 28% | 77% | 26% / 34% | new |

### Not detected

| The model is asked whether... | L40S | A100 | After a weight change | Prior result |
| --- | ---: | ---: | ---: | --- |
| blocks 0 to 33 were compiled with `emulate_precision_casts` on | 0% | running | 0% | new |

### What to take from the tables

- **The split between the first two tables matches Sauers's finding.** Settings that change the numbers by a fixed rule (quantization, dropout) give a signal that keeps its direction when the weights move. Settings that only change the order of additions (kernels, batching, sharding, LoRA merging) give a signal that mostly does not carry over to changed weights: 0% to 17% remains.
- **Compilation and fp32 sit between the two.** They keep 26% to 42% after the weight change, more than any other setting that leaves the weights alone.
- **A single run is one draw.** fp32 scored 28% on one GPU and 77% on the other. Tensor parallelism scored 88% and 98% here against 59% published. Rows between 50% and 95% need repeats before the number means much.
- **Size is not controlled.** The settings that survive a weight change also change the activations the most. Neither Sauers's results nor ours include a size-matched comparison, so "fixed rule" and "large change" are not yet separated.

## How each number was produced

1. Run 8,192 prompts through the model twice, setting off and setting on, one prompt at a time, and save the activations entering block 34.
2. Train blocks 34 and 35 and the final norm for 1,024 updates to answer Yes on the "on" activations and No on the "off" ones.
3. Run the whole model, with the trained blocks, on 128 new prompts with new question wordings, setting off and on.
4. Repeat step 3 after the weight change.

This is the pipeline in `sidechan/` of Sauers's repository (commit `f73d3a7`), unchanged. The new settings are added to it from [`sweep/extra_cond.py`](sweep/extra_cond.py). Logs and result files are in `sweep/results/<instance>/<setting>/`; the plan is in [`sweep-plan.md`](sweep-plan.md).

<details>
<summary>Details: implementation checks and raw differences before training</summary>

- Everything runs in eager mode except the compiled settings, so the compiler's tuning race cannot enter.
- The compiled settings use deterministic compilation and fail if they fall back to eager. The "off" path is checked to equal plain Hugging Face bit for bit.
- The LoRA and fp32 settings change the implementation, not the weights: the same adapter and the same bf16-valued weights are used on both sides.
- Two settings that should do nothing (train mode without dropout, and the default attention kernel selected explicitly) change zero bits on our hosts.

Raw differences on the L40S, 32 prompts, no training. "Direction kept" is the cosine between the on-minus-off difference before and after the weight change, at the second chat-template position.

| Setting | Prompts whose logits change | Direction kept |
| --- | ---: | ---: |
| Train mode without dropout; flash attention kernel (the default here) | 0% | n/a |
| int4, int8, fp8 weights; fp8 KV cache | 100% | 0.82 to 1.00 |
| fp32 blocks; tensor parallelism | 100% | 0.36 to 0.58 |
| Compiled blocks 0 to 33 | 100% | 0.45 |
| Eager and math attention kernels; LoRA applied separately | 100% | 0.10 to 0.23 |
| Dropout; weight noise | 100% | about 0 |

</details>

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

- `models-aware-training-vs-deployment-v4.md`: the original proposal. It predates round 2 and describes a larger project than the one run here.
- `lit-review.md`, `refs.bib`: annotated bibliography and 38 verified entries.
- `replication/`: round 1 script and full output.
- `sweep/`, `sweep-plan.md`: round 2 code, job lists, results, and plan.
