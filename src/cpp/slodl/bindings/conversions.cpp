#include "slodl/bindings/conversions.hpp"

#include <cstddef>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

Tensor as_tensor(const NpArray& array) {
    std::vector<std::size_t> dims;
    dims.reserve(static_cast<std::size_t>(array.ndim()));
    for (py::ssize_t i = 0; i < array.ndim(); i++) {
        dims.push_back(static_cast<std::size_t>(array.shape(i)));
    }

    const double* buffer = array.data();
    return Tensor(
        std::move(dims),
        std::vector<double>(buffer, buffer + static_cast<std::size_t>(array.size())));
}

py::buffer_info as_buffer(Tensor& tensor) {
    const std::vector<std::size_t>& dims = tensor.shape();
    const std::vector<std::size_t>& element_strides = tensor.element_strides();

    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> byte_strides;
    shape.reserve(dims.size());
    byte_strides.reserve(dims.size());
    for (std::size_t i = 0; i < dims.size(); i++) {
        shape.push_back(static_cast<py::ssize_t>(dims[i]));
        // The buffer protocol counts strides in bytes; Tensor counts elements.
        byte_strides.push_back(
            static_cast<py::ssize_t>(element_strides[i] * sizeof(double)));
    }

    return py::buffer_info(
        tensor.data(),
        static_cast<py::ssize_t>(sizeof(double)),
        py::format_descriptor<double>::format(),
        static_cast<py::ssize_t>(dims.size()),
        std::move(shape),
        std::move(byte_strides));
}
