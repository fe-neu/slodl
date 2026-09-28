#include <utility>

#include "slodl/autograd/ops.hpp"
#include "slodl/autograd/functions.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/tensor/ops.hpp"

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
    Tensor result = add_kernel(a, b);

    if (should_record({a, b})) {
        set_history(result, std::make_shared<AddBackward>(), {a, b});
    }
    return result;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return add(a, b);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor result = sub_kernel(a, b);

    if (should_record({a, b})) {
        set_history(result, std::make_shared<SubBackward>(), {a, b});
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
    Tensor result = mul_kernel(a, b);

    if (should_record({a, b})) {
        set_history(result, std::make_shared<MulBackward>(a, b), {a, b});
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return mul(a, b);
}

Tensor sum(const Tensor& a) {
    Tensor result = sum_kernel(a);

    if (should_record({a})) {
        set_history(result, std::make_shared<SumBackward>(), {a});
    }
    return result;
}
