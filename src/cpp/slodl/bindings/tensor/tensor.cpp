#include <cstddef>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::vector <-> list/tuple, for dims and data

#include "slodl/bindings/register.hpp"
#include "slodl/tensor/tensor.hpp"

namespace py = pybind11;

namespace {

// Python allows a negative index counting back from the end; the C++
// operator[] takes an unsigned index, so fold it here.
std::size_t normalize_index(const Tensor& self, py::ssize_t index) {
    const py::ssize_t length = static_cast<py::ssize_t>(
        self.shape().empty() ? 0 : self.shape()[0]);
    if (index < 0) {
        index += length;
    }
    if (index < 0 || index >= length) {
        throw py::index_error("Index out of range");
    }
    return static_cast<std::size_t>(index);
}

}  // namespace

void register_tensor(py::module_& m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<std::size_t>>(), py::arg("dims"))
        .def(py::init<std::vector<std::size_t>, double>(),
             py::arg("dims"), py::arg("init_value"))
        .def(py::init<std::vector<std::size_t>, std::vector<double>>(),
             py::arg("dims"), py::arg("data"))
        .def_property_readonly(
            "shape",
            [](const Tensor& self) { return self.shape(); })
        .def(
            "item",
            [](const Tensor& self) { return self.item(); })
        .def(
            "__len__",
            [](const Tensor& self) {
                if (self.shape().empty()) {
                    throw py::type_error("len() of a 0-dimensional tensor");
                }
                return self.shape()[0];
            })
        // Indexing the last dimension yields a scalar rather than a
        // 0-dimensional Tensor. C++ cannot do this - the return type is fixed
        // at compile time - but Python can, so t[1][0] is a plain float.
        .def(
            "__getitem__",
            [](const Tensor& self, py::ssize_t index) -> py::object {
                Tensor view = self[normalize_index(self, index)];
                if (view.shape().empty()) {
                    return py::float_(view.item());
                }
                return py::cast(std::move(view));
            },
            py::arg("index"))
        // t[i] = 5.0 writes a scalar; t[i] = other copies values in.
        .def(
            "__setitem__",
            [](Tensor& self, py::ssize_t index, double value) {
                Tensor view = self[normalize_index(self, index)];
                view = value;
            },
            py::arg("index"), py::arg("value"))
        .def(
            "__setitem__",
            [](Tensor& self, py::ssize_t index, const Tensor& other) {
                Tensor view = self[normalize_index(self, index)];
                view = other;
            },
            py::arg("index"), py::arg("value"))
        .def("__repr__", &Tensor::repr);
}
