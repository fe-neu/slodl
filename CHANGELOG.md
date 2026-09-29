# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Tensor`: a dense, n-dimensional array of `float64` values, implemented in
  C++17 and exposed through `slodl._core`. In Python it is constructed from
  data, as `Tensor([[1, 2], [3, 4]])`, matching `numpy.array` and
  `torch.tensor`; nested sequences, NumPy arrays, other tensors and bare
  numbers are all accepted, and ragged input is rejected. The C++ class keeps
  its dimension-based constructors.
- `zeros`, `ones` and `full`: create a tensor from a shape rather than from
  data, mirroring their NumPy counterparts.
- `Tensor.clone` (and `Tensor::clone` in the core): a deep copy with its own
  storage, laid out contiguously even when cloning a view.
- Views: indexing a tensor returns a view onto the same storage rather than a
  copy, so writes through a view are visible from the tensor it came from. A
  view keeps its storage alive even after the tensor it came from is gone.
- `Tensor.copy_` and `Tensor.fill_`: write values into a tensor's existing
  storage, so writes through overlapping views of one storage read the source
  before writing. In Python these back `a[0] = b` and `a[0] = 1.0`, which keep
  copying values as they do in NumPy.
- Scalar access: indexing every dimension yields a single element. In C++ that
  is a 0-dimensional tensor, unwrapped with `item()`; in Python it is a plain
  `float`.
- NumPy interoperability: `Tensor(array)` and `Tensor.from_numpy(array)` copy
  an array's contents in, converting dtype and layout as needed, and
  `__array__` exposes a tensor to `np.asarray` through the buffer protocol
  without copying, so the resulting array shares the tensor's memory.
- `repr` prints a tensor's values, nested by dimension and aligned like NumPy.
  Tensors of more than 1000 elements are summarised to the first and last
  three along each dimension, with the full shape appended.
- Typed public API: ships `py.typed` and a hand-written stub for the compiled
  extension.
- Reverse-mode autograd: `requires_grad_`, `requires_grad`, `is_leaf`,
  `grad`, `grad_fn`, `detach` and `backward` on `Tensor`, plus `no_grad` and
  `is_grad_enabled`. Every operation records the node that produced its result;
  `backward()` walks that graph from a 0-dimensional tensor, summing gradients
  where a tensor was used more than once and accumulating them into each leaf.
  Gradients accumulate across passes, as in PyTorch.
- Arithmetic: `add`, `sub`, `mul`, `div` and `neg`, with the `+ - * /` and
  unary `-` operators, all element-wise and differentiable. Plain numbers work
  as operands on either side.
- `matmul` and the `@` operator: matrix multiplication, strictly
  2-dimensional, with no batching and no vector special cases.
- Reductions: `sum` and `mean`, both producing a 0-dimensional tensor, which is
  what `backward()` needs to start from. `mean` is composed from `sum` and
  `div` rather than given a node of its own, so its `grad_fn` reads
  `DivBackward`.
- Broadcasting: operands whose shapes differ are stretched NumPy-style, lining
  shapes up from the right. Stretching is `expand`, a view with a stride of 0
  along each stretched axis, recorded like any other operation, so a broadcast
  operand's gradient is summed back down to its own shape.
- `transpose` and the `T` property: swap two axes, returning a view that shares
  storage. Like `expand`, nothing is copied.
- Test suites for both layers: Catch2 cases under `tests/cpp/` and pytest cases
  under `tests/python/`, mirroring the source layout, including
  finite-difference checks of the gradients of `div` and `matmul`.

### Changed

- Assignment aliases rather than copying. `a = b` now makes `a` another name
  for `b`, sharing its storage and autograd state, matching the copy
  constructor; the value-copying behaviour moved to `copy_`. The previous
  semantics silently discarded a result's autograd history and, because
  containers assign internally, corrupted gradients that were shared between
  branches of a graph.
- `Tensor::operator=(double)` became `fill_`, and now writes every element
  rather than only a 0-dimensional tensor's one, matching `torch.Tensor.fill_`.
  In Python this means `t[0] = 1.0` fills a whole row instead of raising.

### Known limitations

- No activations (`relu`, `exp`, `log`), no powers, and no dimension-wise
  reductions such as `sum(dim=...)`, so no softmax or cross-entropy.
- Nothing above tensors: no `zero_grad`, no optimizers, no layers or modules.
  A training loop has to rebind its parameters each step to clear gradients,
  since in-place updates are not exposed to Python.
- `matmul` is 2-dimensional only: no batched matmul and no vector operands.
- Indexing takes a single integer only. There is no slicing, no multi-axis
  indexing such as `t[1, 2]`, and no `reshape`.
- `backward()` requires a 0-dimensional tensor; it takes no explicit gradient
  argument, and gradients of gradients are not supported.
- Mutating a tensor that a live graph saved (through `copy_`, `fill_` or a
  view) silently corrupts the next backward pass. There is no version counter
  to catch it.
- Some errors surface in Python with a misleading type. A data/shape mismatch
  in the constructor, and calling `item()` on a tensor that has dimensions,
  both raise `IndexError` where `ValueError` would fit better. This is because
  the core throws `std::out_of_range`, which pybind11 maps to `IndexError`.

[Unreleased]: https://github.com/fe-neu/slodl/commits/main
