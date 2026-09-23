# torch-multioutput-alias-guard

此守护工具已迁移到统一的 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) 包中。

请改用 umbrella 包：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run multioutput-alias
```

Python API：

```python
from torch_correctness_guards import safe_aminmax, safe_slogdet
```

原功能现在由 `torch_correctness_guards.guards.multioutput_alias` 维护，共享 CLI 名称为 `multioutput-alias`。

此仓库仅作为历史指针保留，并已归档。
