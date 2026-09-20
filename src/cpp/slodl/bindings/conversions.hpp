#ifndef SLODL_BINDINGS_CONVERSIONS_HPP
#define SLODL_BINDINGS_CONVERSIONS_HPP

// NumPy <-> Tensor marshalling: the only place pybind11 types meet Tensor.

#include <pybind11/numpy.h>

#include "slodl/tensor/tensor.hpp"

// Contiguous, row-major, float64; other layouts/dtypes are cast to fit.
using NpArray =
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

// Copies the array's elements into a new Tensor. Any ndim is accepted,
// including 0-D, since a Tensor has no fixed rank.
Tensor as_tensor(const NpArray& array);

// Describes a Tensor's memory for the buffer protocol; no data is copied.
pybind11::buffer_info as_buffer(Tensor& tensor);

#endif
