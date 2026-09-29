#include <algorithm>
#include <stdexcept>

#include "slodl/tensor/shape.hpp"

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

std::vector<std::size_t> broadcast_shapes(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b
) {
    const std::size_t length = std::max(a.size(), b.size());
    std::vector<std::size_t> result(length, 1);

    // Right-aligned: axis `i` of the result pairs with the axes that many
    // places from the end of each input, and a shorter input simply has none.
    for (std::size_t i = length; i-- > 0;) {
        const std::size_t from_end = length - i;
        const std::size_t left = from_end <= a.size() ? a[a.size() - from_end] : 1;
        const std::size_t right = from_end <= b.size() ? b[b.size() - from_end] : 1;

        if (left != right && left != 1 && right != 1) {
            throw std::invalid_argument(
                "shapes " + format_shape(a) + " and " + format_shape(b) +
                " cannot be broadcast together");
        }
        result[i] = std::max(left, right);
    }
    return result;
}

std::vector<std::size_t> broadcast_strides(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    const std::vector<std::size_t>& target
) {
    if (target.size() < shape.size()) {
        throw std::invalid_argument(
            "shape " + format_shape(shape) + " cannot be read as " +
            format_shape(target) + ": it has more dimensions");
    }

    // Leading axes the shape does not have are read as size 1, so they never
    // move: their stride is 0.
    std::vector<std::size_t> result(target.size(), 0);
    const std::size_t offset = target.size() - shape.size();

    for (std::size_t i = 0; i < shape.size(); i++) {
        const std::size_t dim = shape[i];
        const std::size_t target_dim = target[offset + i];

        if (dim == target_dim) {
            result[offset + i] = strides[i];
        }
        else if (dim == 1) {
            // Stretched: stay on the one element this axis has.
            result[offset + i] = 0;
        }
        else {
            throw std::invalid_argument(
                "shape " + format_shape(shape) + " cannot be read as " +
                format_shape(target));
        }
    }
    return result;
}
