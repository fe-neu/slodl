<p align="center">
  <img src="https://raw.githubusercontent.com/fe-neu/slodl/main/assets/slodl_banner_mono.png"
       alt="slodl" width="640">
</p>

<p align="center">
  <a href="https://pypi.org/project/slodl/"><img
     src="https://img.shields.io/pypi/v/slodl" alt="PyPI"></a>
</p>

# slodl

**Slow deep learning.** A deep-learning framework written from scratch to see
how the pieces actually work — the tensor, autograd, the operations built on
them — rather than to compete with the established frameworks.

It will always be slower than PyTorch or NumPy, which is where the name comes
from. Nothing is hidden behind a library call: the storage, the views, the
computation graph and every derivative are written out in a compiled C++17
core, with a typed Python API on top.

If you want a framework to train real models with, use PyTorch. If you want to
read one end to end, this is meant to be small enough to do that.

## Install

```bash
pip install slodl
```

NumPy is the only runtime dependency. Wheels cover CPython 3.9–3.14 on Linux,
macOS and Windows; building from source needs nothing but a C++17 compiler.

## What you get

- **A tensor that behaves like NumPy's array** — `Tensor([[1, 2], [3, 4]])`
  takes data, `zeros`/`ones`/`full` take a shape, indexing gives views, and
  `repr` prints the values rather than a summary of the object.
- **Gradients** — mark a tensor with `requires_grad_()`, compute a loss, call
  `backward()`, and read `.grad`. Recording can be switched off with
  `no_grad()`, and `detach()` cuts a single tensor loose.
- **The operations to build a layer** — `+ - * / @`, unary `-`, `sum`, `mean`
  and `transpose`, all differentiable, with NumPy-style broadcasting so a bias
  row adds to a whole batch and plain numbers work as operands.
- **NumPy either way** — build a tensor from any array, and hand one to
  `np.asarray` without copying, so the two share memory.
- **Type hints** — ships `py.typed` and stubs, so editors and type checkers
  see the API.

## Quickstart

```python
import numpy as np
from slodl import Tensor

t = Tensor([[1, 2], [3, 4]])
t
# Tensor([[1, 2],
#         [3, 4]])

t.shape        # [2, 2]
t[1][0]        # 3.0  — a float, not a tensor
len(t)         # 2
```

As in NumPy and PyTorch, the argument is the *data*, not the dimensions — so
`Tensor([2, 3])` is a 1-dimensional tensor holding 2.0 and 3.0. Creating by
shape has its own functions:

```python
from slodl import zeros, ones, full

zeros([2, 3])             # 2x3, all 0.0
ones([2, 2])              # 2x2, all 1.0
full([2, 2], 7.0)         # 2x2, all 7.0
Tensor(5.0)               # 0-dimensional; read it with .item()
```

Indexing gives a view, so writing through it changes the original:

```python
row = t[0]
row[1] = 50.0
t[0][1]        # 50.0
```

Assigning a tensor copies its values into the destination, and `clone` gives an
independent tensor when you want one:

```python
t[1] = t[0]        # row 1 now holds row 0's values
copy = t.clone()   # shares nothing with t
```

### Autograd

Mark the tensors you want gradients for, compute a scalar, and call
`backward()`:

```python
from slodl import Tensor

a = Tensor([2.0, 3.0]).requires_grad_()
b = Tensor([10.0, 20.0]).requires_grad_()

(a * b).sum().backward()
a.grad        # Tensor([10, 20])
b.grad        # Tensor([2, 3])
```

`backward()` needs a 0-dimensional tensor, which is what `sum` and `mean` are
for. Gradients accumulate, so a second pass adds to the first.

Matrix multiplication is `@`, and a row broadcasts across a batch, so a linear
layer is one line:

```python
import slodl

inputs = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # batch of 2
weights = slodl.full([3, 2], 0.5).requires_grad_()
bias = slodl.full([2], 0.1).requires_grad_()

outputs = inputs @ weights + bias
outputs.mean().backward()

weights.grad.shape   # [3, 2]
bias.grad.shape      # [2]
```

Recording can be turned off, and a single tensor can be cut loose from its
history:

```python
from slodl import no_grad

with no_grad():
    prediction = inputs @ weights + bias   # computes, records nothing

weights.detach()      # same storage, no history
```

NumPy goes in and out. Building from an array copies; `np.asarray` shares
memory, so writes through the array reach the tensor:

```python
t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

array = np.asarray(t)
array[0, 0] = 99.0
t[0][0]        # 99.0
```

Arrays that are not `float64`, not contiguous, or not row-major — a transpose,
a stepped slice, an `int32` array — are converted on the way in:

```python
Tensor(np.arange(6).reshape(2, 3).T).shape   # [3, 2]
```

## Status

Early and incomplete, but a linear model trains end to end: element-wise
arithmetic, `matmul`, reductions, broadcasting and reverse-mode autograd all
work.

Not there yet: activations (`relu`, `exp`, `log`), dimension-wise reductions
such as `sum(dim=...)`, `reshape` and slicing, and anything above tensors —
no `zero_grad`, optimizers, layers or datasets. `matmul` is 2-dimensional
only, and `backward()` starts from a 0-dimensional tensor.

[CHANGELOG.md](CHANGELOG.md) tracks what has landed and the known limitations.

## Contributing

Building from a checkout, running the test suites and the layout the code
follows are in [CONTRIBUTING.md](CONTRIBUTING.md).
