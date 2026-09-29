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

MatMulBackward::MatMulBackward(const Tensor& a, const Tensor& b): Node("MatMulBackward"), a(a.detach()), b(b.detach()) {}

std::vector<std::optional<Tensor>> MatMulBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    // Only one arrangement has shapes that line up: with a [n, k] and b
    // [k, m], the gradient is [n, m], so the left one must end up [n, k] and
    // the right one [k, m].
    return {
        matmul_kernel(*grad_out[0], b.transpose()),
        matmul_kernel(a.transpose(), *grad_out[0])
    };
}

TransposeBackward::TransposeBackward(std::size_t dim0, std::size_t dim1)
    : Node("TransposeBackward"), dim0(dim0), dim1(dim1) {}

std::vector<std::optional<Tensor>> TransposeBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    return {grad_out[0]->transpose(dim0, dim1)};
}

SumBackward::SumBackward() : Node("SumBackward") {}

std::vector<std::optional<Tensor>> SumBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    return {Tensor(input_shapes[0], grad_out[0]->item())};
}

ExpandBackward::ExpandBackward() : Node("ExpandBackward") {}

std::vector<std::optional<Tensor>> ExpandBackward::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    return {sum_to_size(*grad_out[0], input_shapes[0])};
}
