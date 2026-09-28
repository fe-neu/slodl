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

SubBackward::SubBackward() : Node("SubBackward") {}

std::vector<std::optional<Tensor>> SubBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    return {grad_out[0], neg_kernel(*grad_out[0])};
}

NegBackward::NegBackward() : Node("NegBackward") {}

std::vector<std::optional<Tensor>> NegBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    return {neg_kernel(*grad_out[0])};
}

MulBackward::MulBackward(const Tensor& a, const Tensor& b) : Node("MulBackward"), a(a.detach()), b(b.detach()) {}

std::vector<std::optional<Tensor>> MulBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {

    return {mul_kernel(*grad_out[0], b), mul_kernel(*grad_out[0], a)};
}

DivBackward::DivBackward(const Tensor& a, const Tensor& b) : Node("DivBackward"), a(a.detach()), b(b.detach()) {}

std::vector<std::optional<Tensor>> DivBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {

    return {
        div_kernel(*grad_out[0], b),
        mul_kernel(
            neg_kernel(*grad_out[0]),
            div_kernel(a, mul_kernel(b, b))
        )
    };
}

SumBackward::SumBackward() : Node("SumBackward") {}

std::vector<std::optional<Tensor>> SumBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {

    return {Tensor(input_shapes[0], grad_out[0]->item())};
}