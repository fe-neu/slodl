#ifndef SLODL_BINDINGS_REGISTER_HPP
#define SLODL_BINDINGS_REGISTER_HPP

// One register_* per exposed class, each defined in its own TU, all called
// from _core.cpp. TensorStorage is intentionally absent - it stays internal.

#include <pybind11/pybind11.h>

void register_tensor(pybind11::module_& m);
void register_autograd(pybind11::module_& m);

#endif
