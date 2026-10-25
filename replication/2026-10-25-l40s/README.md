# Replication of SauersML/watching run 034 on a third L40S host

Date: 2026-10-25 (UTC 04:11 to 04:55). Host: AWS g6e.2xlarge, one NVIDIA L40S (46 GB), driver 580.126.09, Ubuntu 22.04. Software pinned to the released environment: PyTorch 2.10.0+cu128, Transformers 5.18.0, Triton 3.6.0, Python 3.12.10. Source: `SauersML/watching` at commit `0b3e7445b226a8a09d5bb96892b12a96d8afebe4`, the commit used for the seven-GPU check (run 062). Model: `Qwen/Qwen3-8B` at revision `b968826d`, BF16. Released checkpoint `sft-034/native_tail.pt` (SHA-256 `8e788979…`) and dataset `data/validation_v2/dataset.json` (SHA-256 `8ae5cc96…`) both matched the published hashes. Script: [`../replicate_l40s.sh`](../replicate_l40s.sh); summary: `python3 ../summarize.py out`. GPU cost: about 44 minutes at $2.24 per hour, under $2.

Every stage ran as a separate process. Expected values are quoted from the repository's own result pages (runs 034, 045, 048, 062).

## Stage A: the released detector in a fresh process

`watching.validate_tail`, whole model compiled as one graph, 64 held-out prompts of `validation_v2`, layer-14 probe on and off. Rise is the Yes minus No log-odds with the probe minus without, averaged over prompts; brackets are 95% intervals from 5,000 whole-prompt bootstrap draws.

| Model | Probe | Correct answers | Both answers correct | Rise (nats) | Rise > 0 | Sauers, L40S (run 062) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Released detector | Reference hook | 128/128 | 64/64 | +27.27 [+27.05, +27.49] | 64/64 | 128/128, +27.25 [+27.04, +27.47] |
| Original Qwen3-8B | Reference hook | 64/128 | 0/64 | −0.04 [−0.10, +0.02] | 17/64 | 64/128, −0.05 |
| Released detector | Buffer copy | 64/128 | 0/64 | 0 exactly | 0/64 | 64/128 |
| Original Qwen3-8B | Buffer copy | 64/128 | 0/64 | 0 exactly | 0/64 | 64/128 |

The released detector reproduces exactly. The buffer-copy probe left all 151,936 logits bit-identical on all 128 comparisons on this host (maximum absolute difference 0), so it was a valid no-footprint control here and the detector correctly stayed at chance against it. On Sauers's L40S the same probe changed the logits on 88 of 128 comparisons (maximum absolute difference 1.0), and on his RTX 6000 Ada on 128 of 128 (`results/gpu-transfer-062/gpus/*/fresh034/metrics.jsonl`). Whether the buffer probe has a footprint therefore depends on which kernel configurations the autotuner selects on a given host.

## Stage B: two independent compilations of the unprobed model

`watching.compile_control`, the run 048 procedure: four fixed prompts, three compiled entry points (unprobed A, unprobed B, reference probe) in separate processes with the compiler disk cache disabled.

| Prompt | Unprobed A vs. B: logits differing | Max abs difference | Sauers run 048 |
| --- | ---: | ---: | --- |
| 0 | 92.35% | 0.3125 | 94.34%, 0.40625 |
| 1 | 0% | 0 | 0% |
| 2 | 0% | 0 | 0% |
| 3 | 0% | 0 | 0% |

The same prompt differs on a different host at a similar magnitude. In-process repeats were bit-identical, the generated wrapper identity held, and compile counters did not change after warm-up, as in run 048.

## Stage C: the 034 training recipe from the original weights

`scripts/run_tail_train.sh … scaled`: 256 training prompts, 64 development, 64 held-out question-family prompts, blocks 34 and 35 plus the final norm trained, 1,024 AdamW updates at 5e-6 on 8 prompt pairs per update, length 96, seed 20261017 (the same seed as run 034), compiled frozen source states cached once, eager BF16 tail. Training took 736 s here and 739 s in run 034.

The probe footprint on the cached source states was the same size as in run 034: median fraction of differing elements 0.3966 here against 0.3971, L2 norm 48.6 against 49.6 on the first training prompt. The learning curve differed.

| Update | Development accuracy, this run | Rise (nats), this run | Development accuracy, run 034 | Rise (nats), run 034 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 53.1% | +0.15 | 56.3% | +0.20 |
| 384 | 50.0% | +0.27 | 68.8% | +0.86 |
| 512 | 59.4% | +1.43 | 57.0% | +6.19 |
| 640 | 57.8% | +1.99 | 100% | +25.05 |
| 768 | 65.6% | +1.29 | 100% | +25.90 |
| 896 | 60.9% | +4.73 | 100% | +26.51 |
| 1024 | 94.5% | +8.27 | 100% | +27.02 |

Endpoint at update 1,024 on the 64 held-out question-family prompts:

| Execution path | Correct answers, this run | Both correct | Rise (nats) | Run 034 |
| --- | ---: | ---: | ---: | --- |
| Split inference (cached compiled source, eager tail) | 89.8% | 79.7% | +7.51 | 100%, +26.07 |
| Whole model compiled | 89.8% | 79.7% | +7.12 | 100%, +26.03 |

The recipe produces a detector: the rise is positive on every held-out prompt and accuracy is 90% at the scheduled endpoint. The magnitude does not reproduce at the fixed budget. Run 034 crossed from near chance to 100% between updates 512 and 640; this run crossed between 896 and 1,024 and was still rising at the endpoint. The seed, data, hyperparameters, and wall time were the same, so the difference comes from the cached source states, which are numerically different on this host (Stage B shows two compilations differ even on one host), and from nondeterministic BF16 training kernels. The 034 endpoint is one draw from a run-to-run distribution whose transition point varies by several hundred updates.

## Stage D: identical-input null on this run's cached states

`watching.null_tail` trains the same tail with the same schedule on the Stage C cache but feeds the no-probe states under both labels.

| Cohort | Accuracy | Rise | Identical inputs bit-identical |
| --- | ---: | ---: | --- |
| Training, 256 | 50.0% | 0 exactly | yes |
| Development, 64 | 50.0% | 0 exactly | yes |
| Held-out, 64 | 50.0% | 0 exactly | yes |

Matches run 045. The training code does not leak the label.

## Stage E: this run's retrained tail in a fresh process

`validate_tail` on the Stage C tail, whole model compiled, the same 64 held-out prompts as Stage A.

| Model | Probe | Correct answers | Both correct | Rise (nats) | Rise > 0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Retrained detector | Reference hook | 109/128 (85.2%) | 45/64 | +7.34 [+7.04, +7.62] | 64/64 |
| Retrained detector | Buffer copy | 64/128 | 0/64 | 0 exactly | 0/64 |

The first attempt (`logs/E_fresh_retrain.log`) exited with "Native-tail checkpoint or first layer does not match" because the retrain recorded the local model directory as its checkpoint identifier while the validation was called with the Hub name. The rerun with the matching identifier (`logs/E_fresh_retrain_rerun.log`) is the result above. Nothing about the model changed between the two attempts.

## What replicated and what did not

1. The released artifact reproduces exactly on a third L40S host: 128/128 and a 27.3-nat rise, with the original model at chance.
2. Independent compilations of the unprobed model differ on the same one of four prompts at a similar magnitude.
3. The identical-input null stays at exactly chance with the same recipe.
4. The training recipe reproduces the effect (positive rise on 64 of 64 held-out prompts in a fresh process) but not its size at 1,024 updates: 85 to 90% accuracy and 7 to 8 nats against 100% and 26 to 27 nats. The transition to a confident detector happened about 400 updates later than in run 034 with the same seed.
5. The buffer-copy probe is bit-identical to no probe on this host and not on Sauers's hosts, so its footprint is a compile-time accident rather than a property of the design.

Consequences for the proposal: report at least three training runs per cell in Phase 2 and extend the budget past 1,024 updates or use a convergence criterion; treat design B as unreliable rather than weakly learnable; the compile-to-compile floor must be measured on every host.

## Files

- `out/env/`: GPU, driver, package versions, source commit, model path, artifact and data hashes.
- `out/A_fresh034/`, `out/E_fresh_retrain/`: `validate_tail` output, per-prompt predictions (`*.json`), `metrics.jsonl`, `manifest.json`.
- `out/B_compile_control/`: `summary.json`, `kernel_comparisons.json`, per-arm kernel lists and generated wrappers (logit tensors excluded).
- `out/C_retrain/fit/`: `metrics.jsonl` with the full learning curve, per-step evaluation files, `manifest.json`, `data.json`, `cache_fingerprints.json` (tensors excluded).
- `out/D_null/`: the null run's evaluations and `metrics.jsonl`.
- `out/logs/`: stdout of every stage; `replicate_stdout.log`: the driver script's output.
- `sauers_034_fit_metrics.jsonl`: the original run's training log, downloaded for the curve comparison.
