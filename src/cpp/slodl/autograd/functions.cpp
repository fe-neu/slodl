#include "slodl/autograd/functions.hpp"
#include "slodl/tensor/ops.hpp"

AddBackward::AddBackward() : Node("AddBackward") {}

std::vector<std::optional<Tensor>> AddBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    // Both inputs get the same gradient, and they may share one tensor:
    // gradients are never written to in place.
    return {grad_out[0], grad_out[0]};
}

MulBackward::MulBackward(const Tensor& a, const Tensor& b) : Node("MulBackward"), a(a.detach()), b(b.detach()) {}

std::vector<std::optional<Tensor>> MulBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {

    return {mul_kernel(*grad_out[0], b), mul_kernel(*grad_out[0], a)};
}

SumBackward::SumBackward() : Node("SumBackward") {}

std::vector<std::optional<Tensor>> SumBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {

    return {Tensor(input_shapes[0], grad_out[0]->item())};
}