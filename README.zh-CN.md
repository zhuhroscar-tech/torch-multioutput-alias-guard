[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-multioutput-alias-guard

检测并守护一类真实存在的 PyTorch 正确性缺陷：接受元组形式 `out=` 参数的多输出操作，
当同一个张量被同时传给元组中的两个不同槽位时，会**静默计算并写入错误结果**，而不是
报错。根本原因：该操作的 meta 函数分别验证每个输出的 dtype/形状，但从未像
`kthvalue`/`cross` 那样针对相同的别名（aliasing）风险调用 `at::assert_no_overlap`。

目前已在本机独立复现确认两个操作存在此问题：

1. **`torch.aminmax(x, out=(t, t))`** —— [pytorch/pytorch#195338](https://github.com/pytorch/pytorch/issues/195338)
   （"torch.aminmax silently returns wrong values when out= tensors alias"），
   被 PyTorch 团队标记为 `module: correctness (silent)`。两次写入竞争同一块内存；
   无论 min 还是 max 最后写入，都会静默覆盖另一个值——别名调用甚至可能返回一个
   算术上不可能的 `min > max` 结果，且没有任何警告。

2. **`torch.linalg.slogdet(x, out=(sign, logabsdet))`**，当 `sign` 与 `logabsdet`
   是同一个张量时——这是本工具自身证据收集过程中的新发现，解答了 #195338 报告者自己
   提出但未能验证的问题（其构建环境缺少 LAPACK，因此无法测试任何 LAPACK 支持的多输出
   操作）。在本机的 Accelerate/LAPACK 构建上，别名调用会静默丢失 `sign` 值，共享张量中
   只剩下 `logabsdet`。

两者是**同一类底层缺陷**，而非两个互不相关的问题——本工具用一个共享的别名检查同时
守护两者，而非拆成两个几乎重复的单操作工具。`var_mean`/`std_mean` 根本不接受元组
`out=`（不适用），`linalg.eigh` 的两个输出形状不兼容，同一张量别名会在触发该缺陷之前
先引发广播错误（不可利用）——详见下方"适用范围"。

## 安装与诊断

需要 Python 3.9+ 及 PyTorch。从源码安装：

```bash
git clone https://github.com/zhuhroscar-tech/torch-multioutput-alias-guard.git
cd torch-multioutput-alias-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-multioutput-alias-guard
torch-multioutput-alias-guard --json
```

如果你已经管理了兼容的 PyTorch 安装，可以不带 extra 直接安装 `.`。使用 `--no-color`
输出纯文本。

退出码：**0** 表示两个守护函数在所有别名测试用例上都成功报错（如果上游缺陷恰好未
复现，会打印"info"提示而非警告）；**1** 表示某个守护函数在至少一个别名用例上未能
报错；**2** 表示无法导入 PyTorch。守护检查成功本身并不意味着上游缺陷已在你的构建上
复现；请单独查看 `any_aminmax_alias_bug` / `any_slogdet_alias_bug`。

## Python API

```python
from torch_multioutput_alias_guard import safe_aminmax, safe_slogdet

# 替代：
#   torch.aminmax(x, out=(t, t))              # 若别名则静默出错
mn, mx = safe_aminmax(x, out=(min_t, max_t))    # 若别名则在计算前报错

# 替代：
#   torch.linalg.slogdet(B, out=(t, t))       # 若别名则静默丢失 sign
sign, logabsdet = safe_slogdet(B, out=(sign_t, logabsdet_t))
```

两个守护函数都会在调用底层操作**之前**，通过 `Tensor.data_ptr()` 相等性（PyTorch 自身
`assert_no_overlap` 使用的同一快速路径检查）验证每一对输出张量参数是否存在别名，若存在
则抛出 `AliasedMultiOutputError`（`RuntimeError` 的子类），错误信息风格与 PyTorch 自身
的重叠错误文本保持一致。当 `out=None`（常见情况——让操作自行分配输出）时，守护函数
是零成本的直接透传：只有调用者显式传入张量时才可能发生别名，因此无需检查。

## 适用范围与局限性

- 仅在 CPU 上复现和测试过（torch 2.14.0，macOS arm64 Accelerate/LAPACK 构建 +
  Ubuntu CI）。未单独验证 CUDA/MPS。
- 仅通过 `data_ptr()` 相等性检测"同一张量对象"级别的别名——这正是两个已确认缺陷的
  具体形态（字面上同一个张量被传入两次）。不尝试实现 PyTorch 更完整的部分重叠分析
  （例如两个不同的张量对象是同一存储在不同偏移处的重叠视图）；这是这两个操作中
  更罕见的调用模式，且未在任一上游报告中观察到。
- `var_mean`/`std_mean` 根本不接受元组 `out=`——不适用本守护。`linalg.eigh` 的特征值/
  特征向量输出形状不兼容，同一张量别名会在触发缺失的 `assert_no_overlap` 路径之前先
  引发广播 `RuntimeError`——不可利用，因此未加以守护。
- `diagnose()` 每次都会针对当前安装的 torch 构建重新运行实际复现实验——绝不假设某个
  特定的 PyTorch 版本受影响或不受影响。

## 开发

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_multioutput_alias_guard --cov-report=term-missing
```

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
