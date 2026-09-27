#ifndef OPS_HPP
#define OPS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "slodl/tensor/tensor.hpp"

/**
 * Number of elements a tensor of this shape holds.
 *
 * @param shape  Dimensions, as returned by Tensor::shape().
 * @return The product of all dimensions; 1 for a 0-dimensional shape, and 0
 *         if any dimension is 0.
 */
std::size_t element_count(const std::vector<std::size_t>& shape);

/**
 * Renders a shape for error messages, e.g. "[2, 3]" or "[]".
 *
 * @param shape  Dimensions, as returned by Tensor::shape().
 * @return The dimensions, comma-separated, in square brackets.
 */
std::string format_shape(const std::vector<std::size_t>& shape);

/**
 * Applies an operation to each pair of logical elements of two tensors.
 *
 * Kernels only: this records no autograd history, so the result is always a
 * leaf that requires no gradient. Either input may be a view, and elements are
 * paired by logical index rather than by position in memory.
 *
 * Every element-wise kernel is built on this, so the iteration lives in one
 * place; broadcasting will be added here rather than in each kernel.
 *
 * @param a          Left operand.
 * @param b          Right operand, which must have exactly the shape of `a`.
 *                   Shapes are not broadcast against each other.
 * @param operation  Callable invoked as operation(double, double) for each
 *                   pair of elements, returning the element of the result.
 * @return A newly allocated, contiguous tensor of the same shape as the
 *         inputs, holding the results.
 * @throws std::invalid_argument if the two shapes differ.
 */
template <typename Operation>
Tensor elementwise(const Tensor& a, const Tensor& b, Operation operation) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            "elementwise: shapes " + format_shape(a.shape()) + " and " +
            format_shape(b.shape()) + " do not match");
    }

    // Walking the inputs by hand: data() points at the view's first element,
    // but the elements after it are strides apart, not adjacent.
    const std::vector<std::size_t>& dims = a.shape();
    Tensor out(dims);

    const std::size_t count = element_count(dims);
    if (count == 0) {
        return out;
    }

    const double* left = a.data();
    const double* right = b.data();
    double* result = out.data();
    const std::vector<std::size_t>& left_strides = a.element_strides();
    const std::vector<std::size_t>& right_strides = b.element_strides();

    // The odometer: `counter` holds the current index per axis and the two
    // offsets follow it, so advancing one element is a pair of additions. When
    // an axis reaches its length it is rewound and the next axis carries.
    std::vector<std::size_t> counter(dims.size(), 0);
    std::size_t left_offset = 0;
    std::size_t right_offset = 0;

    for (std::size_t i = 0; i < count; i++) {
        result[i] = operation(left[left_offset], right[right_offset]);

        for (std::size_t axis = dims.size(); axis-- > 0;) {
            counter[axis]++;
            left_offset += left_strides[axis];
            right_offset += right_strides[axis];
            if (counter[axis] < dims[axis]) {
                break;
            }
            counter[axis] = 0;
            left_offset -= dims[axis] * left_strides[axis];
            right_offset -= dims[axis] * right_strides[axis];
        }
    }
    return out;
}

/**
 * Adds two tensors element by element.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the inputs require gradients. Use the recording `add` for that.
 *
 * @param a  Left operand.
 * @param b  Right operand, which must have exactly the shape of `a`.
 * @return A newly allocated, contiguous tensor holding the sums.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor add_kernel(const Tensor& a, const Tensor& b);

/**
 * Calculates Hadamard Product of two tensors element by element.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the inputs require gradients. Use the recording `mul` for that.
 *
 * @param a  Left operand.
 * @param b  Right operand, which must have exactly the shape of `a`.
 * @return A newly allocated, contiguous tensor holding the Hadamard product.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor mul_kernel(const Tensor& a, const Tensor& b);
#endif
