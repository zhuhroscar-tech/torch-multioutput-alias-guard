# torch-multioutput-alias-guard

This guard has moved into the consolidated [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) package.

Use the umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run multioutput-alias
```

Python API:

```python
from torch_correctness_guards import safe_aminmax, safe_slogdet
```

The original functionality is now maintained as `torch_correctness_guards.guards.multioutput_alias` with the shared CLI name `multioutput-alias`.

This repository is retained only as a historical pointer and is archived.
