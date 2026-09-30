# Contributing

slodl is a solo learning project, so this file is not really a call for
contributions and not a process to follow — it exists to write down the
development setup before I forget it, plus the layout the code happens to use.

That said: fork it, open an issue, send a pull request, or take the code and do
something else with it. Nothing here is precious.

## Development install

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

### VS Code

`.vscode/settings.json` points the CMake Tools extension at `.venv/bin/cmake`
and puts `.venv/bin` on `PATH` for every configure/build/test run, since this
repo's `cmake` and `ninja` come from the venv rather than the system. It also
enables the CTest integration, so the Catch2 cases appear in the Testing view
next to the pytest tests. Select the **dev** preset in the CMake status bar
after opening the folder.

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

## Layout

Each area of the library is one directory under `src/cpp/slodl/`, with its
bindings, Python wrapper, and tests mirroring that layout.

Within `src/cpp/slodl/`, the dependency direction is one-way:

- `tensor/` — storage, the `Tensor` type, shape arithmetic, and the kernels
  that compute values. Kernels know nothing about autograd, so their results
  are always plain leaves.
- `autograd/` — the graph: `Node` and `Edge`, the backward formulas in
  `functions.{hpp,cpp}`, the engine, grad mode, and the recording ops in
  `ops.{hpp,cpp}` that call a kernel and then decide whether to build a node.
- `bindings/` — pybind11 glue only.

An operation therefore lands in three places: a kernel that computes it, a
`Node` subclass holding its derivative, and a recording op that ties them
together.

## Adding a component

1. **C++ core** — `src/cpp/slodl/<area>/<name>.{hpp,cpp}`; add the `.cpp` to
   the `slodl_core` source list in `CMakeLists.txt`. Behaviour belongs here,
   not in the bindings, so that C++ callers and the Catch2 suite can reach it.
   Document declarations in the header with `/** ... */` blocks: what it does,
   `@param`, `@return`, `@throws`. Implementation notes stay in the `.cpp`.
2. **Binding** — `src/cpp/slodl/bindings/<area>/<name>.cpp` defining
   `register_<name>(pybind11::module_&)`; declare it in
   `bindings/register.hpp`, call it from `bindings/_core.cpp`, and add the
   `.cpp` to `pybind11_add_module(_core ...)`. Keep this layer to
   Python-specific adaptation only: negative indices, exception types the
   Python protocols require, and overload dispatch. NumPy marshalling goes in
   `bindings/conversions.{hpp,cpp}`, the one place pybind11 types meet the
   core.
3. **Python** — `src/python/slodl/<area>/_<name>.py` wrapping
   `slodl._core.<Name>` by composition, with NumPy-style docstrings and
   runnable examples. Re-export from `<area>/__init__.py` and the top-level
   `__init__.py`, and add it to `_core.pyi`.
4. **Tests** — `tests/cpp/<area>/test_<name>.cpp` and
   `tests/python/<area>/test_<name>.py`. Every C++ test file in an area shares
   one area tag (`[tensor]`), so a new file needs no change in
   `tests/cpp/CMakeLists.txt`; a new area adds its own `catch_discover_tests`
   line with its own `TEST_PREFIX`.

For a new differentiable operation, the gradient is worth checking against
finite differences rather than against hand-worked arithmetic; `div` and
`matmul` have examples in `tests/cpp/autograd/test_ops.cpp`.
