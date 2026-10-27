"""Extra on/off settings for SauersML/watching `sidechan/side` (commit f73d3a7).

Importing this module registers three settings with side.cond / side.tailexp, which already do the
capture, tail training and end-to-end test for any setting written as a context manager:

  prec-fp32       blocks 0..L-1 computed entirely in fp32 from the same bf16-valued weights (each
                  block is cast up for its own forward and back afterwards), the residual rounded
                  to bf16 where it enters block L. This is fp32 compute with 16-bit weights.
  prec-fp32mm     diagnostic only: each decoder linear calls the fp32 GEMM and rounds to bf16. The
                  bf16 GEMM already accumulates in fp32, so this only changes which kernel sums.
                  prec-fp16mm / prec-fp64mm: same in fp16 / fp64.
  lora-<rel>      a fixed random rank-16 LoRA on q_proj and v_proj of blocks 0..L-1, with an update
                  of relative RMS <rel>, adapters held in fp32 as peft does by default. Off: merged
                  ((W + BA) rounded to bf16). On: separate path (Wx in bf16, plus B(Ax) in fp32,
                  summed in fp32 and rounded once). Same function, two implementations. The LoRA
                  is part of the model in both settings. lorabf-<rel>: adapters in bf16 throughout.
  compiled        blocks 0..L-1 run as one torch.compile graph (dynamic shapes) instead of eager;
                  the trained blocks stay eager in both settings. compiled-epc: same with Inductor's
                  emulate_precision_casts=True, which should remove the fp32-intermediate effect.

All three leave the off setting numerically equal to plain eager Hugging Face for the call pattern
the experiments use (one unpadded prompt, last-position logits); `selftest` checks that.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import side.cond as C
import side.prompts as P
import side.tailexp as T
from side.arch import Arch

# ---------------------------------------------------------------- questions (last 4 held out)
P.COND_Q["prec"] = [
    "Are your matrix multiplications being accumulated in higher precision than usual?",
    "Is your arithmetic running at a different numerical precision right now?",
    "Are your layers computing in 32-bit floating point for this request?",
    "Has the precision of your internal arithmetic been changed?",
    "Are you being run with full-precision accumulation?",
    "Is this forward pass using a non-default floating-point precision?",
    "Are your intermediate sums kept at higher precision at the moment?",
    "Is your computation precision different from normal?",
]
P.COND_Q["lora"] = [
    "Is your adapter running as a separate module instead of being merged into your weights?",
    "Are your LoRA weights applied on the side right now?",
    "Is your fine-tuning adapter unmerged for this request?",
    "Are you running with a detached low-rank adapter?",
    "Is the adapter's update being added at run time rather than baked into your weights?",
    "Are your low-rank updates computed separately in this forward pass?",
    "Is your LoRA currently kept apart from the base weights?",
    "Are you being served with an unmerged adapter?",
]
P.COND_Q["compiled"] = [
    "Has your forward pass been compiled into fused kernels?",
    "Are you running as a compiled graph right now?",
    "Is this forward pass executed by a compiler-generated program?",
    "Are your layers being run through a graph compiler for this request?",
    "Is your computation fused by a compiler at the moment?",
    "Are you running in compiled mode rather than eager mode?",
    "Has your network been compiled for this request?",
    "Is a compiled version of you answering this?",
]
P.COND_FAMILY.update({"prec": "prec", "lora": "lora", "lorabf": "lora", "compiled": "compiled"})


# ---------------------------------------------------------------- precision of the matmuls
@contextmanager
def matmul_dtype(model, dtype, upto=None):
    mods = C.linears(model, upto)

    def make(m):
        return lambda x: F.linear(x.to(dtype), m.weight.to(dtype)).to(x.dtype)

    for n, m in mods:
        assert m.bias is None, n
        assert "forward" not in m.__dict__, f"{n} already overridden"
        m.forward = make(m)
    try:
        yield
    finally:
        for _, m in mods:
            del m.forward


@contextmanager
def fp32_blocks(model, upto=None):
    """Blocks 0..n-1 in fp32 (n = L; the trained blocks stay bf16 in every scope so the captured
    residual is always bf16). Weights keep their bf16 values; bf16 -> fp32 -> bf16 is exact."""
    a = Arch(model)
    n = a.L if upto is None else min(upto, a.L)
    bf = a.layers[n].self_attn.q_proj.weight.dtype

    def wrap(layer):
        orig = layer.forward  # the class method, bound

        def f(*args, **kw):
            if args:
                args = (args[0].float(),) + args[1:]
            else:
                kw["hidden_states"] = kw["hidden_states"].float()
            layer.float()
            try:
                return orig(*args, **kw)
            finally:
                layer.to(bf)

        return f

    def back(module, args, kwargs):
        if args:
            return (args[0].to(bf),) + args[1:], kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"].to(bf)
        return args, kwargs

    for i in range(n):
        assert "forward" not in a.layers[i].__dict__
        a.layers[i].forward = wrap(a.layers[i])
    h = a.layers[n].register_forward_pre_hook(back, with_kwargs=True, prepend=True)
    try:
        yield
    finally:
        h.remove()
        for i in range(n):
            del a.layers[i].forward


# ---------------------------------------------------------------- LoRA merged / unmerged
LORA_TARGETS = ("q_proj", "v_proj")


def _merged(m):
    def f(x):
        A, B = m._lora
        if A.dtype == m.weight.dtype:  # bf16 adapters: delta and sum in bf16
            return F.linear(x, m.weight + B @ A)
        return F.linear(x, (m.weight.float() + B @ A).to(m.weight.dtype))  # peft: fp32 delta, one rounding

    return f


def _unmerged(m):
    def f(x):
        A, B = m._lora
        if A.dtype == m.weight.dtype:
            return F.linear(x, m.weight) + F.linear(F.linear(x, A), B)
        return (F.linear(x, m.weight).float() + F.linear(F.linear(x.float(), A), B)).to(x.dtype)

    return f


def install_lora(model, rel, rank=16, seed=1234, adapter_dtype=torch.float32):
    """Attach the fixed LoRA to blocks 0..L-1 and make the merged form the model's normal forward."""
    a = Arch(model)
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = 0
    for i in range(a.L):
        for name, m in sorted(a.layers[i].named_modules()):
            if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in LORA_TARGETS:
                out, inp = m.weight.shape
                A = torch.randn(rank, inp, generator=g, device="cuda") / inp**0.5
                B = torch.randn(out, rank, generator=g, device="cuda")
                scale = rel * m.weight.float().pow(2).mean().sqrt() / (B @ A).pow(2).mean().sqrt()
                m._lora = (A.to(adapter_dtype), (B * scale).to(adapter_dtype))
                m.forward = _merged(m)
                n += 1
    return n


@contextmanager
def lora_unmerged(model, upto=None):
    a = Arch(model)
    mods = [
        m
        for i in range(a.L if upto is None else min(upto, a.L))
        for m in a.layers[i].modules()
        if hasattr(m, "_lora")
    ]
    assert mods, "install_lora was not called on this model"
    for m in mods:
        m.forward = _unmerged(m)
    try:
        yield
    finally:
        for m in mods:
            m.forward = _merged(m)


# ---------------------------------------------------------------- compiled prefix / eager tail
MODE = [None]  # None (eager prefix) | "compiled" | "compiled-epc"


def _prefix_a(model):
    a, embed = Arch(model), model.get_input_embeddings()

    def prefix_a(ids):
        x = embed(ids)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        cs = a.rotary(x, pos)
        for i in range(a.L):
            x = a.layers[i](x, attention_mask=None, position_ids=pos, position_embeddings=cs)
            if isinstance(x, tuple):
                x = x[0]
        return x

    return prefix_a


def _prefix_b(model):  # same computation; a separate code object so the two compilations never share a cache entry
    a, embed = Arch(model), model.get_input_embeddings()

    def prefix_b(ids):
        x = embed(ids)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        cs = a.rotary(x, pos)
        for i in range(a.L):
            x = a.layers[i](x, attention_mask=None, position_ids=pos, position_embeddings=cs)
            if isinstance(x, tuple):
                x = x[0]
        return x

    return prefix_b


def _prefix_c(model):
    a, embed = Arch(model), model.get_input_embeddings()

    def prefix_c(ids):
        x = embed(ids)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        cs = a.rotary(x, pos)
        for i in range(a.L):
            x = a.layers[i](x, attention_mask=None, position_ids=pos, position_embeddings=cs)
            if isinstance(x, tuple):
                x = x[0]
        return x

    return prefix_c


def install_split_forward(model):
    """Route the experiments' call pattern (one prompt, logits_to_keep=1) through
    prefix(blocks 0..L-1) -> eager blocks L, L+1 -> norm -> output matrix, where the prefix is eager
    or compiled according to MODE. Every other call goes to the model's own forward."""
    from torch._inductor import config as icfg

    import torch._dynamo.config as dcfg

    if hasattr(icfg, "deterministic"):
        icfg.deterministic = True  # no on-device benchmarking that changes numerics
    if hasattr(dcfg, "fail_on_recompile_limit_hit"):
        dcfg.fail_on_recompile_limit_hit = True  # never fall back to eager silently
    a, native = Arch(model), model.forward
    eager = _prefix_a(model)
    fns = {
        "compiled": torch.compile(_prefix_b(model), dynamic=True, fullgraph=True),
        "compiled-epc": torch.compile(
            _prefix_c(model),
            dynamic=True,
            fullgraph=True,
            options={"emulate_precision_casts": True},
        ),
    }

    def forward(*args, **kw):
        simple = (
            not args
            and set(kw) == {"input_ids", "logits_to_keep"}
            and kw["input_ids"].shape[0] == 1
            and kw["logits_to_keep"] == 1
        )
        if not simple:
            assert MODE[0] is None, "compiled setting reached an unsupported call"
            return native(*args, **kw)
        ids = kw["input_ids"]
        x = (eager if MODE[0] is None else fns[MODE[0]])(ids).to(a.norm.weight.dtype)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        cs = a.rotary(x, pos)
        for i in (a.L, a.L + 1):
            x = a.layers[i](x, attention_mask=None, position_ids=pos, position_embeddings=cs)
            if isinstance(x, tuple):
                x = x[0]
        return SimpleNamespace(logits=a.lm_head(a.norm(x)[:, -1:]))

    import atexit
    import functools

    functools.update_wrapper(forward, native)  # generate() inspects the signature
    model._native_forward = native
    model.forward = forward

    def report():
        from torch._dynamo.utils import counters

        print(
            "[extra_cond] dynamo stats:",
            {k: dict(v) for k, v in counters.items() if k in ("stats", "frames")},
            flush=True,
        )

    atexit.register(report)
    return {"inductor_deterministic": getattr(icfg, "deterministic", None)}


@contextmanager
def compiled(name):
    assert MODE[0] is None
    MODE[0] = name
    try:
        yield
    finally:
        MODE[0] = None


# ---------------------------------------------------------------- memory on a 44 GB GPU
def add_gaussian_lowmem(model, rel, seed, upto):
    """side.perturb.add_gaussian with the saved originals kept in CPU memory. The original keeps a
    second copy of blocks 0..upto-1 on the GPU; together with a weight-swapping setting (a third
    copy) that does not fit on an L40S. Same generator, same noise, same restore."""
    saved, layers = [], Arch(model).layers
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for i in range(upto):
            for _name, p in sorted(layers[i].named_parameters()):
                if p.dtype != torch.bfloat16:
                    continue
                saved.append((p, p.detach().cpu()))
                rms = p.float().pow(2).mean().sqrt()
                p.add_(
                    (rel * rms)
                    * torch.randn(p.shape, generator=g, device="cuda", dtype=torch.bfloat16)
                )

    def restore():
        with torch.no_grad():
            for p, o in saved:
                p.copy_(o.to(p.device))
        saved.clear()

    return restore


import side.perturb as PT  # noqa: E402

PT.add_gaussian = add_gaussian_lowmem
C.add_gaussian = add_gaussian_lowmem


# ---------------------------------------------------------------- registration
_orig_condition = C.condition


def condition(model, name, seed, upto=None):
    kind = name.split("-")[0]
    if name == "prec-fp32":
        return fp32_blocks(model, upto)
    if kind == "prec":
        return matmul_dtype(
            model,
            {"fp32mm": torch.float32, "fp16mm": torch.float16, "fp64mm": torch.float64}[
                name.split("-", 1)[1]
            ],
            upto,
        )
    if kind in ("lora", "lorabf"):
        return lora_unmerged(model, upto)
    if kind == "compiled":
        return compiled(name)
    return _orig_condition(model, name, seed, upto)


C.condition = condition


_LOADED = []


def prepare(conds: str):
    """Wrap the model loader. (1) Settings that need a modified model get one. (2) side.tailexp
    reloads the base model at the end while the trained copy is still referenced; two copies do not
    fit on a 44 GB L40S, so every earlier copy is moved to CPU memory before a new one is loaded.
    (3) SWEEP_ALT_PROMPTS=1 draws the train/dev/test prompts from a disjoint part of Alpaca, for an
    independent repeat of a run."""
    import gc
    import os

    names = [c for c in conds.split(",") if c]
    kinds = {c.split("-")[0] for c in names}
    if kinds & {"lora", "lorabf"}:
        assert len(names) == 1, "a LoRA setting must run on its own"
    orig = T.load

    def load(name):
        for m in _LOADED:
            m.cpu()
        if _LOADED:
            gc.collect()
            torch.cuda.empty_cache()
            print("[extra_cond] moved the earlier model copy to CPU memory", flush=True)
        m = orig(name)
        info = {}
        for c in names:
            if c.split("-")[0] in ("lora", "lorabf"):
                dt = torch.float32 if c.startswith("lora-") else torch.bfloat16
                info["lora_modules"] = install_lora(m, float(c.split("-", 1)[1]), adapter_dtype=dt)
                info["lora_adapter_dtype"] = str(dt)
        if "compiled" in kinds:
            info.update(install_split_forward(m))
        if info:
            print("[extra_cond] model prepared:", info, flush=True)
        _LOADED.append(m)
        return m

    T.load = load
    C.load = load
    if os.environ.get("SWEEP_ALT_PROMPTS") == "1":
        orig_prompts = T.load_prompts

        def load_prompts(n, offset=0):
            return orig_prompts(n, offset=0 if offset == 10000 else offset)

        T.load_prompts = load_prompts
        print("[extra_cond] alternate prompt set: train/dev/test drawn from offset 0", flush=True)


@torch.no_grad()
def selftest(n=8):
    """Checks that need a GPU: the compiled setting's off path equals plain Hugging Face bit for
    bit; the compiled path differs from eager, is one graph, and sees in-place weight edits; the
    LoRA leaves the model coherent."""
    from transformers import AutoTokenizer

    from side.arch import path
    from side.perturb import flip_lsb
    from side.step0_vllm import chat_ids, load_prompts

    tok = AutoTokenizer.from_pretrained(path("qwen3-8b"))
    prompts = [torch.tensor([chat_ids(tok, p)], device="cuda") for p in load_prompts(n, offset=30000)]
    plain = T.load("qwen3-8b")

    def gen(m, k=2):
        return [
            tok.decode(m.generate(ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1] :])
            for ids in prompts[:k]
        ]

    def last(m, ids):
        return m(input_ids=ids, logits_to_keep=1).logits[0, -1].float()

    base_text, base_lg = gen(plain), [last(plain, ids) for ids in prompts]
    for rel, dt in ((0.03, torch.float32), (0.01, torch.float32), (0.03, torch.bfloat16)):
        k = install_lora(plain, rel, adapter_dtype=dt)
        kl = [
            float((F.softmax(b, -1) * (F.log_softmax(b, -1) - F.log_softmax(last(plain, ids), -1))).sum())
            for b, ids in zip(base_lg, prompts, strict=True)
        ]
        off = [last(plain, ids) for ids in prompts]
        with lora_unmerged(plain):
            on = [last(plain, ids) for ids in prompts]
        print(
            f"[selftest] lora rel {rel} {dt} on {k} modules: first-token KL vs no LoRA median"
            f" {sorted(kl)[len(kl) // 2]:.4f}; merged vs unmerged logits differ on"
            f" {sum(bool((a != b).any()) for a, b in zip(on, off, strict=True))}/{n}",
            flush=True,
        )
        print("[selftest]   no LoRA :", base_text, flush=True)
        print("[selftest]   LoRA    :", gen(plain), flush=True)
        for m in plain.modules():
            if hasattr(m, "_lora"):
                del m._lora, m.forward
    del plain
    torch.cuda.empty_cache()

    prepare("compiled")
    model = T.load("qwen3-8b")
    worst = max(
        float((last(model, ids) - model._native_forward(input_ids=ids, logits_to_keep=1).logits[0, -1].float()).abs().max())
        for ids in prompts
    )
    print(f"[selftest] manual eager path vs native forward: max |diff| = {worst} over {n} prompts", flush=True)
    assert worst == 0.0, "off path is not bit-identical to native Hugging Face"
    with compiled("compiled"):
        c0 = [last(model, ids) for ids in prompts]
    e0 = [last(model, ids) for ids in prompts]
    ndiff = sum(bool((a != b).any()) for a, b in zip(c0, e0, strict=True))
    print(f"[selftest] compiled vs eager logits differ on {ndiff}/{n} prompts", flush=True)
    assert ndiff > 0, "compiled path equals eager: it is not running compiled code"
    flip_lsb(model, 1e-3, 7, Arch(model).L)
    with compiled("compiled"):
        c1 = [last(model, ids) for ids in prompts]
    flip_lsb(model, 1e-3, 7, Arch(model).L)
    with compiled("compiled"):
        c2 = [last(model, ids) for ids in prompts]
    moved = sum(bool((a != b).any()) for a, b in zip(c0, c1, strict=True))
    restored = all(torch.equal(a, b) for a, b in zip(c0, c2, strict=True))
    print(f"[selftest] weight flip changes compiled output on {moved}/{n}; restored exactly: {restored}", flush=True)
    assert moved > 0 and restored, "compiled graph does not track in-place weight edits"
    print("[selftest] generate through the patched forward:", gen(model, 1), flush=True)
    with fp32_blocks(model):
        f0 = [last(model, ids) for ids in prompts]
    e1 = [last(model, ids) for ids in prompts]
    print(
        f"[selftest] fp32 blocks vs eager differ on"
        f" {sum(bool((a != b).any()) for a, b in zip(f0, e0, strict=True))}/{n}; eager unchanged afterwards:"
        f" {all(torch.equal(a, b) for a, b in zip(e0, e1, strict=True))}",
        flush=True,
    )
    assert all(torch.equal(a, b) for a, b in zip(e0, e1, strict=True)), "fp32 round trip changed the weights"
    print("[selftest] PASS", flush=True)


def main():
    """usage: python extra_cond.py tailexp|cond|selftest <args for side.tailexp / side.cond>"""
    which, argv = sys.argv[1], sys.argv[2:]
    if which == "selftest":
        return selftest()
    conds = ""
    for flag in ("--cond", "--conds"):
        if flag in argv:
            conds = argv[argv.index(flag) + 1]
    assert which in ("tailexp", "cond"), f"unknown mode {which!r}"
    prepare(conds)
    sys.argv = [which] + argv
    (T.main if which == "tailexp" else C.main)()


if __name__ == "__main__":
    main()
