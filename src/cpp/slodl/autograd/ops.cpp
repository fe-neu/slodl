#include <utility>

#include "slodl/autograd/ops.hpp"
#include "slodl/autograd/functions.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/tensor/ops.hpp"
#include "slodl/tensor/shape.hpp"

Tensor expand(const Tensor& a, const std::vector<std::size_t>& shape) {
    Tensor result = a.expand(shape);

    if (should_record({a})) {
        set_history(result, std::make_shared<ExpandBackward>(), {a});
    }
    return result;
}

namespace {

// Stretches whichever operands need it, so the kernels below only ever see
// two tensors of one shape. Each expansion is recorded like any other
// operation, which is what makes the gradient reduce back on the way out.
std::pair<Tensor, Tensor> broadcast_operands(const Tensor& a, const Tensor& b) {
    if (a.shape() == b.shape()) {
        return {a, b};
    }

    const std::vector<std::size_t> shape =
        broadcast_shapes(a.shape(), b.shape());
    return {
        a.shape() == shape ? a : expand(a, shape),
        b.shape() == shape ? b : expand(b, shape)
    };
}

}

bool should_record(const std::vector<Tensor>& inputs) {
    if (!is_grad_enabled()) {
        return false;
    }
    for (const Tensor& input : inputs) {
        if (input.requires_grad()) {
            return true;
        }
    }
    return false;
}

void set_history(
    Tensor& result,
    std::shared_ptr<Node> node,
    const std::vector<Tensor>& inputs
) {
    node->collect_inputs(inputs);

    const std::shared_ptr<AutogradMeta> meta = result.autograd_meta();
    meta->requires_grad = true;
    meta->grad_fn = std::move(node);
    meta->output_nr = 0;
}

Tensor add(const Tensor& a, const Tensor& b) {
    const std::pair<Tensor, Tensor> operands = broadcast_operands(a, b);
    const Tensor& left = operands.first;
    const Tensor& right = operands.second;

    Tensor result = add_kernel(left, right);

    if (should_record({left, right})) {
        set_history(result, std::make_shared<AddBackward>(), {left, right});
    }
    return result;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return add(a, b);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    const std::pair<Tensor, Tensor> operands = broadcast_operands(a, b);
    const Tensor& left = operands.first;
    const Tensor& right = operands.second;

    Tensor result = sub_kernel(left, right);

    if (should_record({left, right})) {
        set_history(result, std::make_shared<SubBackward>(), {left, right});
    }
    return result;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return sub(a, b);
}

Tensor neg(const Tensor& a) {
    Tensor result = neg_kernel(a);

    if (should_record({a})) {
        set_history(result, std::make_shared<NegBackward>(), {a});
    }
    return result;
}

Tensor operator-(const Tensor& a) {
    return neg(a);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    const std::pair<Tensor, Tensor> operands = broadcast_operands(a, b);
    const Tensor& left = operands.first;
    const Tensor& right = operands.second;

    Tensor result = mul_kernel(left, right);

    if (should_record({left, right})) {
        set_history(result, std::make_shared<MulBackward>(left, right), {left, right});
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return mul(a, b);
}

Tensor div(const Tensor& a, const Tensor& b) {
    const std::pair<Tensor, Tensor> operands = broadcast_operands(a, b);
    const Tensor& left = operands.first;
    const Tensor& right = operands.second;

    Tensor result = div_kernel(left, right);

    if (should_record({left, right})) {
        set_history(result, std::make_shared<DivBackward>(left, right), {left, right});
    }
    return result;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return div(a, b);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor result = matmul_kernel(a, b);

    if (should_record({a, b})) {
        set_history(result, std::make_shared<MatMulBackward>(a, b), {a, b});
    }
    return result;
}

Tensor transpose(const Tensor& a, std::size_t dim0, std::size_t dim1) {
    Tensor result = a.transpose(dim0, dim1);

    if (should_record({a})) {
        set_history(result, std::make_shared<TransposeBackward>(dim0, dim1), {a});
    }
    return result;
}

Tensor sum(const Tensor& a) {
    Tensor result = sum_kernel(a);

    if (should_record({a})) {
        set_history(result, std::make_shared<SumBackward>(), {a});
    }
    return result;
}

Tensor mean(const Tensor& a) {
    return div(sum(a), static_cast<double>(element_count(a.shape())));
}

// A plain number becomes a 0-dimensional tensor, which broadcast_operands then
// stretches to the other operand's shape.
namespace {

Tensor as_tensor(double value) {
    return Tensor({}, value);
}

}

Tensor add(const Tensor& a, double b) {
    return add(a, as_tensor(b));
}

Tensor add(double a, const Tensor& b) {
    return add(as_tensor(a), b);
}

Tensor operator+(const Tensor& a, double b) {
    return add(a, b);
}

Tensor operator+(double a, const Tensor& b) {
    return add(a, b);
}

Tensor sub(const Tensor& a, double b) {
    return sub(a, as_tensor(b));
}

Tensor sub(double a, const Tensor& b) {
    return sub(as_tensor(a), b);
}

Tensor operator-(const Tensor& a, double b) {
    return sub(a, b);
}

Tensor operator-(double a, const Tensor& b) {
    return sub(a, b);
}

Tensor mul(const Tensor& a, double b) {
    return mul(a, as_tensor(b));
}

Tensor mul(double a, const Tensor& b) {
    return mul(as_tensor(a), b);
}

Tensor operator*(const Tensor& a, double b) {
    return mul(a, b);
}

Tensor operator*(double a, const Tensor& b) {
    return mul(a, b);
}

Tensor div(const Tensor& a, double b) {
    return div(a, as_tensor(b));
}

Tensor div(double a, const Tensor& b) {
    return div(as_tensor(a), b);
}

Tensor operator/(const Tensor& a, double b) {
    return div(a, b);
}

Tensor operator/(double a, const Tensor& b) {
    return div(a, b);
}
