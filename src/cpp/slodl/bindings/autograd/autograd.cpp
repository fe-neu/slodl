#include <memory>

#include <pybind11/pybind11.h>

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

    m.def("add", &add, py::arg("a"), py::arg("b"));
    m.def("mul", &mul, py::arg("a"), py::arg("b"));
    m.def("sub", &sub, py::arg("a"), py::arg("b"));
    m.def("neg", &neg, py::arg("a"));
    // A lambda, not &div: <cstdlib> also declares std::div.
    m.def("div", [](const Tensor& a, const Tensor& b) { return div(a, b); },
          py::arg("a"), py::arg("b"));
    m.def("sum", &sum, py::arg("a"));

    m.def("is_grad_enabled", &is_grad_enabled);
    m.def("set_grad_enabled", &set_grad_enabled, py::arg("enabled"));
}
