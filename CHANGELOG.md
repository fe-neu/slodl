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
- Value-copying assignment: assigning one tensor to another copies elements
  into the destination's existing storage instead of making it refer to the
  source's storage, and requires matching shapes. Copies between overlapping
  views of one storage read the source before writing.
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
- Test suites for both layers: Catch2 cases under `tests/cpp/tensor/` and
  pytest cases under `tests/python/tensor/`, mirroring the source layout.

### Known limitations

- Tensors support no arithmetic yet: no elementwise operations, no matrix
  multiplication, no reductions. Convert with `np.asarray` in the meantime.
- Indexing takes a single integer only. There is no slicing, no multi-axis
  indexing such as `t[1, 2]`, and no `reshape` or `transpose`.
- Some errors surface in Python with a misleading type. A data/shape mismatch
  in the constructor, and calling `item()` on a tensor that has dimensions,
  both raise `IndexError` where `ValueError` would fit better. This is because
  the core throws `std::out_of_range`, which pybind11 maps to `IndexError`.
- A tensor variable cannot be pointed at different storage once created, since
  assignment copies values. `clone` covers making an independent copy, but
  there is no `rebind`.

[Unreleased]: https://github.com/fe-neu/slodl/commits/main
