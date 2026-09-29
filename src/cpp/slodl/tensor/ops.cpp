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

Tensor matmul_kernel(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument(
            "matmul: both operands must be 2-dimensional, got " +
            format_shape(a.shape()) + " and " + format_shape(b.shape()));
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "matmul: shapes " + format_shape(a.shape()) + " and " +
            format_shape(b.shape()) + " do not line up: " +
            std::to_string(a.shape()[1]) + " columns against " +
            std::to_string(b.shape()[0]) + " rows");
    }

    const std::size_t rows = a.shape()[0];
    const std::size_t columns = b.shape()[1];
    const std::size_t shared = a.shape()[1];

    Tensor out({rows, columns});

    // Both operands may be views, so every element is reached through its own
    // strides rather than by walking memory. The output is freshly allocated
    // and therefore contiguous, so it is written straight through.
    const double* left = a.data();
    const double* right = b.data();
    double* result = out.data();
    const std::vector<std::size_t>& left_strides = a.element_strides();
    const std::vector<std::size_t>& right_strides = b.element_strides();

    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < columns; j++) {
            double sum = 0.0;
            for (std::size_t k = 0; k < shared; k++) {
                sum += left[i * left_strides[0] + k * left_strides[1]] *
                       right[k * right_strides[0] + j * right_strides[1]];
            }
            result[i * columns + j] = sum;
        }
    }
    return out;
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

Tensor sum_kernel(const Tensor& a) {
    const double total = reduce_all(a, 0.0, [](double accumulated, double element) {
        return accumulated + element;
    });
    return Tensor({}, total);
}

