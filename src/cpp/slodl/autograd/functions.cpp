#include "slodl/autograd/functions.hpp"

AddBackward::AddBackward() {
    name = "AddBackward";
}

std::vector<std::optional<Tensor>> AddBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    // Both inputs get the same gradient, and they may share one tensor:
    // gradients are never written to in place.
    return {grad_out[0], grad_out[0]};
}
