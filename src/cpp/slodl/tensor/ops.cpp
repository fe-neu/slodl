#include "slodl/tensor/ops.hpp"

std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::string format_shape(const std::vector<std::size_t>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); i++) {
        out += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return out + "]";
}

Tensor add_kernel(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double left, double right) {
        return left + right;
    });
}
