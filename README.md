<p align="center">
  <img src="https://raw.githubusercontent.com/fe-neu/slodl/main/assets/slodl_banner_mono.png"
       alt="slodl" width="640">
</p>

<p align="center">
  <a href="https://pypi.org/project/slodl/"><img
     src="https://img.shields.io/pypi/v/slodl.svg" alt="PyPI"></a>
</p>

# slodl

**Slow deep learning.** A deep-learning framework written from scratch to see
how the pieces actually work — the tensor, autograd, the operations built on
them — rather than to compete with the established frameworks.

It will always be slower than PyTorch or NumPy, which is where the name comes
from. Nothing here is hidden behind a library call: the storage, the views, the
computation graph and every derivative are written out in a compiled C++17
core, with a thin, typed Python API on top. The numeric work lives in an
extension module (`slodl._core`); the Python layer only adapts it to Python
conventions and documents it.

## Features

- **`Tensor`** — a dense, n-dimensional array of `float64` values, laid out
  row-major.
- **Views, not copies** — indexing returns a view onto the same storage, so
  writing through a view is visible from the tensor it came from. Storage is
  reference-counted and outlives any tensor that points into it.
- **Scalars where you expect them** — indexing every dimension gives a plain
  Python `float`, the way NumPy behaves, even though the C++ side has to model
  it as a 0-dimensional tensor.
- **Arithmetic with autograd** — `+ - * / @`, `sum`, `mean`, `transpose`, and
  `neg`, each recording what it did so `backward()` can walk back through it.
- **Broadcasting** — shapes are stretched NumPy-style, so a bias row adds to a
  whole batch and plain numbers work as operands. Stretching is a view with a
  stride of 0, and its gradient is summed back down.
- **Writes are explicit** — assignment makes two names for one tensor, while
  `copy_` and `fill_` write into existing storage. In Python, `a[0] = b` writes
  values, as it does in NumPy.
- **NumPy interoperability** — build a tensor from any array, and hand a tensor
  to `np.asarray` without copying, so the two share memory.
- **Creation that reads like NumPy** — `Tensor([[1, 2], [3, 4]])` takes nested
  data, while `zeros`, `ones` and `full` build from a shape.
- **Readable `repr`** — prints the actual values, nested and aligned like
  NumPy, and summarises anything over 1000 elements.
- **Typed** — ships `py.typed` and stubs.

## Install

```bash
pip install slodl
```

NumPy is pulled in as a runtime dependency. No system CMake, Ninja, or compiler
setup is required beyond a C++17 compiler — scikit-build-core fetches CMake and
Ninja into an isolated build environment automatically.

## Usage

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

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install scikit-build-core pybind11 cmake ninja
pip install --no-build-isolation -e .
```

With the editable install, `pyproject.toml` sets `editable.rebuild = true`, so
editing a `.cpp`/`.hpp`/`CMakeLists.txt` triggers a recompile on the next
`import slodl` — no reinstall, just restart the Python process (or the notebook
kernel).

### Gotcha: editable rebuilds need a real, activated toolchain

`editable.rebuild = true` re-invokes `cmake` and `ninja` at import time. Two
things must hold, or every import after the first fails:

1. **Install with `--no-build-isolation`** (as above). A plain isolated
   `pip install -e .` records a path to CMake inside a temporary build
   environment (`/tmp/pip-build-env-.../cmake`); that directory is deleted
   after the install, so the rebuild step then fails with `cmake: not found`.
   Installing without isolation makes it use the `cmake`/`ninja` from the venv
   instead, which persist.

2. **Activate the venv** (`source .venv/bin/activate`) before running Python or
   starting the notebook kernel, so `.venv/bin` is on `PATH` and the
   import-time rebuild can find `cmake`. Running `.venv/bin/python` directly,
   without activation, is not enough. For a Jupyter kernel you cannot launch
   from an activated shell, add
   `"env": {"PATH": "/abs/path/to/.venv/bin:${PATH}"}` to its `kernel.json`
   instead.

A third way in is to run an isolated build — `pip install .` or `pip wheel .` —
in a checkout that also has an editable install. Both share the `build/` tree,
and the isolated build rewrites it with paths into a temporary environment that
is deleted afterwards, so the next `import slodl` fails the same way.

If an editable checkout gets into a broken state, `rm -rf build` and re-run the
`pip install --no-build-isolation -e .` step.

## Testing

**C++ (Catch2).** Kept out of the wheel build; enabled by the `dev` preset,
which also skips the Python extension so no pybind11 needs to be in scope.
Catch2 is fetched via `FetchContent` on the first configure.

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Without presets: `cmake -S . -B build-test -DSLODL_BUILD_TESTS=ON
-DSLODL_BUILD_PYTHON=OFF && cmake --build build-test && ctest --test-dir
build-test --output-on-failure`.

**Python (pytest).**

```bash
pip install --no-build-isolation -e '.[test]'
pytest
```

The docstrings carry runnable examples, which are not part of the default run:

```bash
pytest --doctest-modules --pyargs slodl
```

## Adding a component

Each area of the library is one directory under `src/cpp/slodl/`, with its
bindings, Python wrapper, and tests mirroring that layout.

1. **C++ core** — `src/cpp/slodl/<area>/<name>.{hpp,cpp}`; add the `.cpp` to
   the `slodl_core` source list in `CMakeLists.txt`. Behaviour belongs here,
   not in the bindings, so that C++ callers and the Catch2 suite can reach it.
2. **Binding** — `src/cpp/slodl/bindings/<area>/<name>.cpp` defining
   `register_<name>(pybind11::module_&)`; declare it in
   `bindings/register.hpp`, call it from `bindings/_core.cpp`, and add the
   `.cpp` to `pybind11_add_module(_core ...)`. Keep this layer to
   Python-specific adaptation only: negative indices, exception types the
   Python protocols require, and overload dispatch. NumPy marshalling goes in
   `bindings/conversions.{hpp,cpp}`, the one place pybind11 types meet the
   core.
3. **Python** — `src/python/slodl/<area>/_<name>.py` wrapping
   `slodl._core.<Name>` by composition, with NumPy-style docstrings. Re-export
   the class from `<area>/__init__.py` and the top-level `__init__.py`, and add
   it to `_core.pyi`.
4. **Tests** — `tests/cpp/<area>/test_<name>.cpp` and
   `tests/python/<area>/test_<name>.py`. Every C++ test file in an area shares
   one area tag (`[tensor]`), so a new file needs no change in
   `tests/cpp/CMakeLists.txt`; a new area adds its own `catch_discover_tests`
   line with its own `TEST_PREFIX`.

## Status

Early and incomplete, but a linear model trains end to end: element-wise
arithmetic, `matmul`, reductions, broadcasting and reverse-mode autograd all
work. Still missing are activations, dimension-wise reductions, `reshape` and
slicing, and anything above tensors — no `zero_grad`, optimizers or layers yet.
See [CHANGELOG.md](CHANGELOG.md) for what has landed and the known limitations.
