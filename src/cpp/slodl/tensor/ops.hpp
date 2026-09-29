#ifndef OPS_HPP
#define OPS_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "slodl/tensor/shape.hpp"
#include "slodl/tensor/tensor.hpp"

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

    const std::vector<std::size_t>& dims = a.shape();
    Tensor out(dims);

    const double* left = a.data();
    const double* right = b.data();
    double* result = out.data();

    for_each_offset<2>(
        dims,
        {&a.element_strides(), &b.element_strides()},
        [&](std::size_t i, const std::array<std::size_t, 2>& offsets) {
            result[i] = operation(left[offsets[0]], right[offsets[1]]);
        });
    return out;
}

/**
 * Applies an operation to each logical element of a tensor.
 *
 * Kernels only: this records no autograd history, so the result is always a
 * leaf that requires no gradient. The input may be a view, whose elements are
 * walked by their strides rather than straight through memory.
 *
 * The one-tensor counterpart to elementwise(), on which every single-operand
 * kernel is built.
 *
 * @param a          Tensor to transform.
 * @param operation  Callable invoked as operation(double) for each element,
 *                   returning the element of the result.
 * @return A newly allocated, contiguous tensor of the same shape as `a`,
 *         holding the results.
 */
template <typename Operation>
Tensor unary_elementwise(const Tensor& a, Operation operation) {
    const std::vector<std::size_t>& dims = a.shape();
    Tensor out(dims);

    const double* element = a.data();
    double* result = out.data();

    for_each_offset<1>(
        dims,
        {&a.element_strides()},
        [&](std::size_t i, const std::array<std::size_t, 1>& offsets) {
            result[i] = operation(element[offsets[0]]);
        });
    return out;
}

/**
 * Folds every logical element of a tensor into a single value.
 *
 * Walks the tensor by its strides, like elementwise(), so a view reduces over
 * the elements it actually describes.
 *
 * @param a          Tensor to reduce.
 * @param initial    Value the fold starts from, and the result for an empty
 *                   tensor.
 * @param operation  Callable invoked as operation(accumulated, element),
 *                   returning the new accumulated value.
 * @return The accumulated value.
 */
template <typename Operation>
double reduce_all(const Tensor& a, double initial, Operation operation) {
    double accumulated = initial;
    const double* elements = a.data();

    for_each_offset<1>(
        a.shape(),
        {&a.element_strides()},
        [&](std::size_t, const std::array<std::size_t, 1>& offsets) {
            accumulated = operation(accumulated, elements[offsets[0]]);
        });
    return accumulated;
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
 * Subtracts two tensors element by element.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the inputs require gradients. Use the recording `sub` for that.
 *
 * @param a  Left operand.
 * @param b  Right operand, subtracted from `a`, which must have exactly the
 *           shape of `a`.
 * @return A newly allocated, contiguous tensor holding the differences.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor sub_kernel(const Tensor& a, const Tensor& b);

/**
 * Flips the sign of every element of a tensor.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the input requires a gradient. Use the recording `neg` for that.
 *
 * @param a  Tensor to negate, which may be a view.
 * @return A newly allocated, contiguous tensor holding the negated elements.
 */
Tensor neg_kernel(const Tensor& a);

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

/**
 * Divides two tensors element by element.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the inputs require gradients. Use the recording `div` for that.
 *
 * @param a  Left operand, the dividend.
 * @param b  Right operand, the divisor, which must have exactly the shape of
 *           `a`. Dividing by zero follows IEEE 754 and yields an infinity, or
 *           a NaN for 0/0, rather than throwing.
 * @return A newly allocated, contiguous tensor holding the quotients.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor div_kernel(const Tensor& a, const Tensor& b);

/**
 * Multiplies two matrices.
 *
 * Strictly 2-dimensional: no batching, and no vector special cases. Each
 * element of the result is the dot product of a row of `a` with a column of
 * `b`.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the inputs require gradients. Use the recording `matmul` for that.
 *
 * @param a  Left operand, of shape [n, k]; may be a view.
 * @param b  Right operand, of shape [k, m]; may be a view.
 * @return A newly allocated, contiguous tensor of shape [n, m].
 * @throws std::invalid_argument if either operand is not 2-dimensional, or if
 *         `a`'s columns do not match `b`'s rows.
 */
Tensor matmul_kernel(const Tensor& a, const Tensor& b);

/**
 * Sums a tensor back down to a shape it was broadcast from.
 *
 * The reverse of stretching: every element of `a` is added into the element of
 * the result it was read from, so an axis that was stretched is summed away
 * and leading axes disappear entirely.
 *
 * A kernel: it records no autograd history.
 *
 * @param a      Tensor to reduce, whose shape `shape` must broadcast to.
 * @param shape  Shape to reduce to.
 * @return A newly allocated, contiguous tensor of `shape` holding the sums,
 *         or a copy of `a` when the shapes already match.
 * @throws std::invalid_argument if `shape` does not broadcast to `a`'s shape.
 */
Tensor sum_to_size(const Tensor& a, const std::vector<std::size_t>& shape);

/**
 * Adds up every element of a tensor.
 *
 * A kernel: it records no autograd history, so the result is a leaf even when
 * the input requires a gradient. Use the recording `sum` for that.
 *
 * @param a  Tensor to add up, which may be a view.
 * @return A 0-dimensional tensor holding the total; zero for an empty tensor.
 */
Tensor sum_kernel(const Tensor& a);
#endif
