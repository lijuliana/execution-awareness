# Plan for the factor sweep (revision 2, after review)

Goal: find out which ways of running Qwen3-8B a model fine-tuned on its last two blocks can tell apart from the arithmetic alone. One L40S per job. Not a scaling or generalization study.

## What changed after the review

Sauers merged `sidechan/` into `SauersML/watching` on 2026-10-26 (commit `f73d3a7`). It already trains detectors for int4 and int8 weights, fp8 KV cache, KV eviction, dropout, weight noise, batch size, tensor parallelism, and eager attention, each with 8,192 prompts, and reports whether each detector survives a low-bit flip of the upstream weights. Its pipeline (`side.tailexp`) takes any on/off setting written as a context manager, runs in eager Hugging Face on unpadded prompts, and does capture, training, and the end-to-end test in one command.

Consequences:

- We reuse that pipeline. The first plan's pipeline did not exist.
- Their settings are replications for us. New work is the settings they did not run.
- Dropped from the first plan: the 128/128 gate on every host (the released detector fails on other GPU types by design), the linear screen (their linear refit reached 32% where the tail reached 97%), 256 prompts (they used 8,192), and the claim that eager vLLM equals Hugging Face (it does not).

## Wave 1 (running)

| Job | Setting | Status in their repo | Flags |
| --- | --- | --- | --- |
| rate | Buffer hook and re-compilation, repeated in fresh processes, default and deterministic compile | Not measured as a rate | `watching.validate_tail` x10, `watching.compile_control` x12 |
| diag | Step-0 difference for every setting, old and new | Published for theirs | `side.cond`, 32 prompts |
| w-int8 | int8 weights | 99% | as published |
| batch | Shares a batch with other prompts | 90% | as published |
| attn-eager | Eager attention vs. SDPA | 59% | as published |
| tp-4 | Tensor parallelism, 4 shards | 59% | as published |
| dropout-0.1 | Dropout active | 100% | as published |
| w-fp8 | fp8 weights | Setting exists, no result | new |
| prec-fp32mm | Matmuls accumulated in fp32, bf16 weights | Not present | new, `sweep/extra_cond.py` |
| lora-0.03 | LoRA merged into weights vs. applied separately | Not present | new |
| compiled | Blocks 0 to 33 as one torch.compile graph vs. eager | Not present | new |

Every training job reports, from their code: the untrained model's answers, the fraction of positions that differ, the learning curve on held-out development prompts, end-to-end accuracy with both answers right on 128 held-out prompts with held-out question wordings, and the same after a low-bit flip and after Gaussian noise on the upstream weights.

## Design decisions

1. **Eager mode for everything except `compiled` and the rate job.** No compiler, so no tuning race.
2. **`compiled` uses Inductor's deterministic mode** and compiles once per process with dynamic shapes; training and testing happen in that process.
3. **The rate job runs alone on its GPU**, each compilation in a fresh process with caches off.
4. **Published flags for replications**, so numbers are comparable. New settings use the same sizes.
5. **Off path checked against plain Hugging Face.** The compiled setting's eager path must equal the model's own forward bit for bit before anything runs.
6. **One run per setting in this wave.** Our replication showed one run is one draw. Anything between 60% and 95% gets repeated with a fresh capture before it is reported as a number.
7. **Nulls come from their code.** `train-mode` and `kv-noop` must change zero bits on our host; the diag job checks this.

## Deferred

- Engines (vLLM, SGLang). Their run 064 needs its own capture code, unpadded prompts, and 13,256 pairs. SGLang needs different PyTorch and Transformers versions.
- GPU type on GCP. After wave 1 shows which settings are worth repeating.
- Cross-scoring each detector on other settings' captures. Needs the trained weights kept, about 0.9 GB per run.

## Operations

- Every instance has a hard shutdown timer and terminates on shutdown.
- A local watcher pulls results every four minutes, terminates each instance when its queue ends, and lists anything still running.
- Budget cap for the sweep: $300.
