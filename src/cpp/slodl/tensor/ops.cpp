#include "slodl/tensor/ops.hpp"

Tensor add_kernel(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double left, double right) {
        return left + right;
    });
}

Tensor sub_kernel(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double left, double right) {
        return left - right;
    });
}

Tensor neg_kernel(const Tensor& a) {
    return unary_elementwise(a, [](double element) {
        return -element;
    });
}

Tensor mul_kernel(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double left, double right) {
        return left * right;
    });
}

Tensor div_kernel(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double left, double right) {
        return left / right;
    });
}


Tensor sum_kernel(const Tensor& a) {
    const double total = reduce_all(a, 0.0, [](double accumulated, double element) {
        return accumulated + element;
    });
    return Tensor({}, total);
}

Tensor sum_to_size(const Tensor& a, const std::vector<std::size_t>& shape) {
    if (a.shape() == shape) {
        return a.clone();
    }

    Tensor out(shape, 0.0);
    double* totals = out.data();
    const double* elements = a.data();

    const std::vector<std::size_t> out_strides =
        broadcast_strides(shape, out.element_strides(), a.shape());

    for_each_offset<2>(
        a.shape(),
        {&a.element_strides(), &out_strides},
        [&](std::size_t, const std::array<std::size_t, 2>& offsets) {
            totals[offsets[1]] += elements[offsets[0]];
        });
    return out;
}
