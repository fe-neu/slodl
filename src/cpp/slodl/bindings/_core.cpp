// The package's single PYBIND11_MODULE; body is just register_* calls.

#include <pybind11/pybind11.h>

#include "slodl/bindings/register.hpp"

PYBIND11_MODULE(_core, m) {
    m.doc() = "Compiled core for slodl. Internal; use the slodl package.";

    register_autograd(m);
    register_tensor(m);
}
