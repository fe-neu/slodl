#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <array>
#include <cstddef>
#include <string>
#include <vector>

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
 * The shape two operands are stretched to when they differ.
 *
 * Shapes are lined up from the right. Two dimensions are compatible if they
 * are equal or one of them is 1, and the longer shape's leading dimensions are
 * taken as they are. This is the NumPy rule.
 *
 * @param a  First shape.
 * @param b  Second shape.
 * @return The combined shape, taking the larger of each pair of dimensions.
 * @throws std::invalid_argument if a pair of dimensions is neither equal nor
 *         has a 1 in it.
 */
std::vector<std::size_t> broadcast_shapes(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b
);

/**
 * Strides that read a tensor of `shape` as though it had `target` elements.
 *
 * A dimension of 1 that is stretched gets a stride of 0, so walking it reads
 * the same element again instead of moving; dimensions the shape does not have
 * at all are prepended the same way. Nothing is copied.
 *
 * @param shape    The tensor's own shape.
 * @param strides  The tensor's own strides, one per dimension of `shape`.
 * @param target   The shape to read it as, at least as long as `shape`.
 * @return Strides with one entry per dimension of `target`.
 * @throws std::invalid_argument if `shape` cannot be stretched to `target`.
 */
std::vector<std::size_t> broadcast_strides(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    const std::vector<std::size_t>& target
);

/**
 * Visits every logical index of a shape, tracking an offset per tensor.
 *
 * The one place that knows how to walk strided data. `body` is called once per
 * element with the flat index and the current offset into each tensor, in the
 * order their strides were given.
 *
 * Offsets are advanced like a mechanical odometer: the innermost axis moves by
 * its stride, and when an axis reaches its length it is rewound and the next
 * one carries. A stride of 0 therefore keeps returning the same element, which
 * is how a broadcast tensor is read.
 *
 * @param dims     The shape being walked.
 * @param strides  One stride vector per tensor, each with `dims.size()`
 *                 entries, so pass broadcast_strides() results for tensors
 *                 whose own shape is shorter.
 * @param body     Callable invoked as body(flat_index, offsets).
 */
template <std::size_t N, typename Body>
void for_each_offset(
    const std::vector<std::size_t>& dims,
    const std::array<const std::vector<std::size_t>*, N>& strides,
    Body body
) {
    const std::size_t count = element_count(dims);
    if (count == 0) {
        return;
    }

    std::array<std::size_t, N> offsets{};
    std::vector<std::size_t> counter(dims.size(), 0);

    for (std::size_t i = 0; i < count; i++) {
        body(i, offsets);

        for (std::size_t axis = dims.size(); axis-- > 0;) {
            counter[axis]++;
            for (std::size_t tensor = 0; tensor < N; tensor++) {
                offsets[tensor] += (*strides[tensor])[axis];
            }
            if (counter[axis] < dims[axis]) {
                break;
            }
            counter[axis] = 0;
            for (std::size_t tensor = 0; tensor < N; tensor++) {
                offsets[tensor] -= dims[axis] * (*strides[tensor])[axis];
            }
        }
    }
}

#endif
