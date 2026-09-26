#include "slodl/autograd/grad_mode.hpp"

namespace {

// Per thread, so a thread that disables recording cannot affect another.
thread_local bool grad_enabled = true;

}

bool is_grad_enabled() {
    return grad_enabled;
}

void set_grad_enabled(bool enabled) {
    grad_enabled = enabled;
}

NoGradGuard::NoGradGuard() : previous(grad_enabled) {
    grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    grad_enabled = previous;
}
