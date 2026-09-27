#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::vector <-> list/tuple, for dims and data

#include "slodl/autograd/autograd.hpp"
#include "slodl/autograd/ops.hpp"
#include "slodl/bindings/conversions.hpp"
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
    // py::buffer_protocol() lets NumPy read a Tensor's memory without copying.
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer(&as_buffer)
        .def_static("from_numpy", &as_tensor, py::arg("array"))
        .def(py::init<std::vector<std::size_t>>(), py::arg("dims"))
        .def(py::init<std::vector<std::size_t>, double>(),
             py::arg("dims"), py::arg("init_value"))
        .def(py::init<std::vector<std::size_t>, std::vector<double>>(),
             py::arg("dims"), py::arg("data"))
        .def_property_readonly(
            "shape",
            [](const Tensor& self) { return self.shape(); })
        .def("clone", &Tensor::clone)
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
        // Both write into the slice: Tensor assignment aliases instead, which
        // is not what t[i] = x means in Python.
        .def(
            "__setitem__",
            [](Tensor& self, py::ssize_t index, double value) {
                Tensor view = self[normalize_index(self, index)];
                view.fill_(value);
            },
            py::arg("index"), py::arg("value"))
        .def(
            "__setitem__",
            [](Tensor& self, py::ssize_t index, const Tensor& other) {
                Tensor view = self[normalize_index(self, index)];
                view.copy_(other);
            },
            py::arg("index"), py::arg("value"))
        .def("__add__", &add, py::arg("other"))
        .def("__mul__", &mul, py::arg("other"))
        // Autograd. requires_grad is a property, like in PyTorch, and
        // requires_grad_ returns nothing: the Python layer returns its own
        // wrapper so that chaining stays on the Python object.
        .def_property(
            "requires_grad",
            &Tensor::requires_grad,
            [](Tensor& self, bool flag) { self.requires_grad_(flag); })
        .def(
            "requires_grad_",
            [](Tensor& self, bool flag) { self.requires_grad_(flag); },
            py::arg("flag") = true)
        .def_property_readonly("is_leaf", &Tensor::is_leaf)
        // A copy of the gradient, which shares its storage, so writing to it
        // writes through to the gradient itself.
        .def_property_readonly(
            "grad",
            [](const Tensor& self) -> std::optional<Tensor> {
                const Tensor* gradient = self.grad();
                if (gradient == nullptr) {
                    return std::nullopt;
                }
                return *gradient;
            })
        .def_property_readonly(
            "grad_fn",
            [](const Tensor& self) { return self.autograd_meta()->grad_fn; })
        .def("detach", &Tensor::detach)
        .def("backward", &Tensor::backward)
        .def("__repr__", &Tensor::repr);
}
