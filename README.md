[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-multioutput-alias-guard

Detect and guard a real PyTorch correctness bug **class**: multi-output
operations whose `out=` parameter accepts a tuple of tensors silently
compute and write a **wrong** result when the SAME tensor is passed for
two different slots of that tuple, instead of raising. Root cause: the
op's meta function validates each output's dtype/shape independently but
never calls `at::assert_no_overlap` the way `kthvalue`/`cross` already do
for the identical aliasing hazard.

Confirmed on two ops so far, both independently reproduced on this host:

1. **`torch.aminmax(x, out=(t, t))`** — [pytorch/pytorch#195338](https://github.com/pytorch/pytorch/issues/195338)
   ("torch.aminmax silently returns wrong values when out= tensors
   alias"), labeled `module: correctness (silent)` by the PyTorch team.
   The two writes race for the same memory location; whichever value
   (min or max) is written last silently overwrites the other — the
   aliased call can even return an arithmetically impossible `min > max`
   pair with zero warning.

2. **`torch.linalg.slogdet(x, out=(sign, logabsdet))`** when `sign` and
   `logabsdet` are the SAME tensor — a new finding from this tool's own
   evidence-gathering, resolving the #195338 issue author's own stated
   open question (their build lacked LAPACK, so they could not test any
   LAPACK-backed multi-output op). On this host's Accelerate/LAPACK
   build, the aliased call silently drops the `sign` value entirely and
   leaves only `logabsdet` in the shared tensor.

Both are the **same underlying defect class**, not two unrelated bugs —
this tool guards both with one shared aliasing check rather than shipping
two near-duplicate single-op tools. `var_mean`/`std_mean` do not accept a
tuple `out=` at all (not applicable), and `linalg.eigh`'s two outputs have
incompatible shapes so same-tensor aliasing raises a broadcast error
before reaching the defect (not exploitable there) — see Scope below.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build. From source:

```bash
git clone https://github.com/zhuhroscar-tech/torch-multioutput-alias-guard.git
cd torch-multioutput-alias-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-multioutput-alias-guard
torch-multioutput-alias-guard --json
```

If you already manage a compatible PyTorch installation, install `.`
without the extra. Use `--no-color` for plain text output.

Exit codes: **0** means both guards raised for every aliased test case
(and prints "info" lines, not warnings, for cases where the underlying
bug also happened not to reproduce), **1** means a guard failed to raise
for at least one aliased case, and **2** means PyTorch could not be
imported. A successful guard check does not by itself mean the upstream
bug was reproduced on your build; inspect `any_aminmax_alias_bug` /
`any_slogdet_alias_bug` separately.

## Python API

Wrap any call site that might pass the same tensor for two output slots:

```python
from torch_multioutput_alias_guard import safe_aminmax, safe_slogdet

# instead of:
#   torch.aminmax(x, out=(t, t))            # silently wrong if aliased
mn, mx = safe_aminmax(x, out=(min_t, max_t))  # raises before computing if aliased

# instead of:
#   torch.linalg.slogdet(B, out=(t, t))     # silently loses sign if aliased
sign, logabsdet = safe_slogdet(B, out=(sign_t, logabsdet_t))
```

Both guards validate that every pair of output tensor arguments does not
alias (via `Tensor.data_ptr()` identity — the same fast-path check
PyTorch's own `assert_no_overlap` uses) **before** calling the underlying
op, and raise `AliasedMultiOutputError` (a `RuntimeError` subclass) with a
message styled after PyTorch's own overlap error text when they do. When
`out=None` (the common case — letting the op allocate its own outputs),
the guard is a zero-cost passthrough: aliasing can only happen when the
caller explicitly passes tensors, so there is nothing to check.

## Scope and limitations

- Reproduced and tested on CPU only (torch 2.14.0, macOS arm64
  Accelerate/LAPACK build + Ubuntu CI). Not separately verified on
  CUDA/MPS.
- Detects same-tensor-OBJECT aliasing via `data_ptr()` equality — the
  exact shape of both confirmed bugs (literally the same tensor passed
  twice). It does not attempt PyTorch's fuller partial-overlap analysis
  (e.g. two different tensor objects that are overlapping views of the
  same storage at different offsets); that is a strictly rarer call
  pattern for these two ops and was not observed in either upstream
  report.
- `var_mean`/`std_mean` do not accept a tuple `out=` at all — not
  applicable to this guard. `linalg.eigh`'s eigenvalue/eigenvector
  outputs have incompatible shapes, so same-tensor aliasing raises a
  broadcast `RuntimeError` before reaching the missing-`assert_no_overlap`
  path — not exploitable there, so not guarded.
- `diagnose()` always re-runs the actual reproduction against whatever
  torch build is installed — it never assumes a specific PyTorch version
  is or isn't affected.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_multioutput_alias_guard --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).
