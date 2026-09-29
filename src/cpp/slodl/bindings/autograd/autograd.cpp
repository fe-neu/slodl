#include <cstddef>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "slodl/autograd/autograd.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/autograd/ops.hpp"
#include "slodl/bindings/register.hpp"

namespace py = pybind11;

void register_autograd(py::module_& m) {
    // Nodes are held by shared_ptr in the graph, so pybind11 must use the same
    // holder. Only the name is exposed: a node is something to look at while
    // debugging, not something to build from Python.
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def_property_readonly("name", [](const Node& self) { return self.name; })
        .def("__repr__", [](const Node& self) {
            return "<" + self.name + ">";
        });

    // Each op has scalar overloads too, so the tensor-to-tensor one has to be
    // picked out explicitly.
    using BinaryOp = Tensor (*)(const Tensor&, const Tensor&);
    m.def("add", static_cast<BinaryOp>(&add), py::arg("a"), py::arg("b"));
    m.def("mul", static_cast<BinaryOp>(&mul), py::arg("a"), py::arg("b"));
    m.def("sub", static_cast<BinaryOp>(&sub), py::arg("a"), py::arg("b"));
    m.def("neg", &neg, py::arg("a"));
    m.def("div", static_cast<BinaryOp>(&div), py::arg("a"), py::arg("b"));
    m.def("sum", &sum, py::arg("a"));
    m.def("mean", &mean, py::arg("a"));
    m.def("transpose",
          [](const Tensor& a, std::size_t dim0, std::size_t dim1) {
              return transpose(a, dim0, dim1);
          },
          py::arg("a"), py::arg("dim0") = 0, py::arg("dim1") = 1);
    m.def("expand",
          [](const Tensor& a, std::vector<std::size_t> shape) {
              return expand(a, shape);
          },
          py::arg("a"), py::arg("shape"));

    m.def("is_grad_enabled", &is_grad_enabled);
    m.def("set_grad_enabled", &set_grad_enabled, py::arg("enabled"));
}
