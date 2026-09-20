# slodl

A slow deep-learning framework for educational purposes, with a compiled C++17
core and a thin, typed Python API. The data structures and numeric work live in
an extension module (`slodl._core`); the public Python layer only adapts them to
Python conventions and documents them.

The project is built from the ground up — starting with the tensor — so the
point is to see how the pieces work, not to compete with NumPy or PyTorch on
speed.

## Features

- **`Tensor`** — a dense, n-dimensional array of `float64` values, laid out
  row-major.
- **Views, not copies** — indexing returns a view onto the same storage, so
  writing through a view is visible from the tensor it came from. Storage is
  reference-counted and outlives any tensor that points into it.
- **Scalars where you expect them** — indexing every dimension gives a plain
  Python `float`, the way NumPy behaves, even though the C++ side has to model
  it as a 0-dimensional tensor.
- **Assignment copies values** — `a[0] = b` writes `b`'s elements into `a`'s
  existing storage rather than quietly rebinding a temporary, which would look
  identical and do nothing.
- **NumPy interoperability** — build a tensor from any array, and hand a tensor
  to `np.asarray` without copying, so the two share memory.
- **Readable `repr`** — prints the actual values, nested and aligned like
  NumPy, and summarises anything over 1000 elements.
- **Typed** — ships `py.typed` and stubs.

## Install

```bash
pip install .
```

NumPy is pulled in as a runtime dependency. No system CMake, Ninja, or compiler
setup is required beyond a C++17 compiler — scikit-build-core fetches CMake and
Ninja into an isolated build environment automatically.

## Usage

```python
import numpy as np
from slodl import Tensor

t = Tensor([2, 2], [1, 2, 3, 4])
t
# Tensor([[1, 2],
#         [3, 4]])

t.shape        # [2, 2]
t[1][0]        # 3.0  — a float, not a tensor
len(t)         # 2
```

Other ways to build one:

```python
Tensor([2, 3])            # zero-filled
Tensor([2, 2], 7.0)       # filled with a value
Tensor([], 5.0)           # 0-dimensional; read it with .item()
```

Indexing gives a view, so writing through it changes the original:

```python
row = t[0]
row[1] = 50.0
t[0][1]        # 50.0
```

Assigning a tensor copies its values into the destination:

```python
t[1] = t[0]    # row 1 now holds row 0's values
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

One way into that broken state is to run an isolated build — `pip install .` or
`pip wheel .` — in a checkout that also has an editable install. Both share the
`build/` tree, and the isolated build rewrites it with paths into a temporary
environment that is deleted afterwards, so the next `import slodl` fails with
`cmake: not found`.

If an editable checkout gets into a broken state, `rm -rf build` and re-run the
`pip install --no-build-isolation -e .` step.

### VS Code

`.vscode/settings.json` points the CMake Tools extension at `.venv/bin/cmake`
and puts `.venv/bin` on `PATH` for every configure/build/test run, since this
repo's `cmake` and `ninja` come from the venv rather than the system. It also
enables the CTest integration, so the Catch2 cases appear in the Testing view
as a `tensor/` tree next to the pytest tests. Select the **dev** preset in the
CMake status bar after opening the folder.

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

Early and incomplete. `Tensor` exists with views, NumPy interoperability, and
element access; there is no arithmetic, no slicing, no `reshape`/`transpose`,
and nothing built on top of tensors yet. See [CHANGELOG.md](CHANGELOG.md) for
what has landed and the current known limitations.
