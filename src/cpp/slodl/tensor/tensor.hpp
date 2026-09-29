#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <memory>

#include "slodl/tensor/tensor_storage.hpp"


class Node;
struct AutogradMeta;

/**
 * A multi-dimensional array of doubles over shared storage.
 *
 * A tensor is a view onto a TensorStorage: an offset, a shape and a stride per
 * axis. Several tensors can therefore describe different parts, or different
 * layouts, of the same buffer, and indexing produces such a view rather than a
 * copy. Use clone() for an independent copy.
 *
 * Copying a Tensor aliases it: the copy shares the storage and the same
 * autograd state, so both names refer to one tensor.
 */
class Tensor {
private:
    std::shared_ptr<TensorStorage> storage;
    std::size_t start_offset;
    std::vector<std::size_t> strides;
    std::vector<std::size_t> dims;

    std::shared_ptr<AutogradMeta> meta;

    static std::size_t get_size_for_dims(std::vector<std::size_t> dims);
    static std::vector<std::size_t> get_strides_for_dims(std::vector<std::size_t> dims);

    std::size_t get_offset_for_flat_index(std::size_t flat_index) const;

    /** Creates a view onto existing storage, with its own autograd state. */
    Tensor(
        std::shared_ptr<TensorStorage> storage,
        std::size_t start_offset,
        std::vector<std::size_t> strides,
        std::vector<std::size_t> dims
    );

public:
    /**
     * Creates a zero-filled tensor.
     *
     * @param dims  Length per axis; empty for a 0-dimensional (scalar) tensor.
     */
    Tensor(std::vector<std::size_t> dims);

    /**
     * Creates a tensor with every element set to the same value.
     *
     * @param dims        Length per axis.
     * @param init_value  Value written to every element.
     */
    Tensor(std::vector<std::size_t> dims, double init_value);

    /**
     * Creates a tensor from values in row-major order.
     *
     * @param dims  Length per axis.
     * @param data  Exactly as many values as the shape holds.
     * @throws std::out_of_range if the number of values does not match.
     */
    Tensor(std::vector<std::size_t> dims, std::vector<double> data);

    /** @return The length of each axis; empty for a scalar tensor. */
    const std::vector<std::size_t>& shape() const;

    /** @return The stride of each axis, counted in elements, not bytes. */
    const std::vector<std::size_t>& element_strides() const;

    /**
     * Pointer to this tensor's first element.
     *
     * Only the first element is at a known position: the rest are
     * element_strides() apart, and a view's elements need not be adjacent or
     * in index order. Walk them with the strides, not by incrementing.
     */
    double* data();
    const double* data() const;

    /**
     * Indexes the first axis.
     *
     * @param index  Position along axis 0.
     * @return A view sharing this tensor's storage, with the first axis
     *         dropped, carrying no autograd history of its own.
     * @throws std::out_of_range if the tensor is 0-dimensional or the index is
     *         out of bounds.
     */
    Tensor operator[](std::size_t index) const;

    /** Aliases `other`: shares its storage and its autograd state. */
    Tensor(const Tensor& other) = default;

    /**
     * Reads this tensor as though it had a larger shape.
     *
     * Dimensions of 1 are stretched to the requested size, and dimensions the
     * tensor does not have are added on the left. Nothing is copied: the
     * stretched axes are given a stride of 0, so the same element is read
     * again for every position along them.
     *
     * @param shape  Shape to read this tensor as, which this tensor's shape
     *               must broadcast to.
     * @return A view sharing this tensor's storage, with the requested shape
     *         and no autograd history of its own.
     * @throws std::invalid_argument if this tensor's shape does not broadcast
     *         to `shape`.
     */
    Tensor expand(const std::vector<std::size_t>& shape) const;

    /**
     * Reads this tensor with two of its axes swapped.
     *
     * Nothing is copied: the view has those axes' lengths and strides
     * exchanged, so the same storage is walked in a different order. The
     * result is generally not contiguous; use clone() for a compact copy.
     *
     * @param dim0  First axis to swap, the outermost by default.
     * @param dim1  Second axis to swap, the next one by default, so that
     *              transpose() alone flips a matrix. Swapping an axis with
     *              itself is a no-op.
     * @return A view sharing this tensor's storage, with no autograd history
     *         of its own.
     * @throws std::out_of_range if either axis is not a dimension of this
     *         tensor.
     */
    Tensor transpose(std::size_t dim0 = 0, std::size_t dim1 = 1) const;

    /** Aliases `other`, like the copy constructor. To write values into this
     * tensor's existing elements, use copy_(). */
    Tensor& operator=(const Tensor& other) = default;

    /**
     * Copies `other`'s values into this tensor's existing elements, in place.
     *
     * Unlike assignment this writes data rather than aliasing, so it also
     * writes through to anything viewing the same storage. Overlapping storage
     * is handled. It records no autograd history.
     *
     * @param other  Tensor of exactly this tensor's shape.
     * @return This tensor.
     * @throws std::invalid_argument if the shapes differ.
     */
    Tensor& copy_(const Tensor& other);

    /**
     * Copies this tensor into fresh, contiguous storage.
     *
     * @return An independent tensor of the same shape and values, with no
     *         autograd history.
     */
    Tensor clone() const;

    /**
     * Reads the single value of a scalar tensor.
     *
     * @return The one element held by a 0-dimensional tensor.
     * @throws std::out_of_range if the tensor has any axes.
     */
    double item() const;

    /**
     * Writes the same value to every element, in place.
     *
     * @param value  Value to store.
     * @return This tensor.
     */
    Tensor& fill_(double value);

    /**
     * Renders the values for display.
     *
     * @return The values, nested by axis; tensors over 1000 elements are
     *         abbreviated with "..." and get a trailing shape.
     */
    std::string repr() const;

    // Autograd:

    /** @return Whether operations on this tensor are recorded for autograd. */
    bool requires_grad() const;

    /**
     * Turns gradient tracking on or off, in place.
     *
     * Only leaves can be changed: a tensor produced by a recorded operation
     * already inherits its status from that operation's inputs.
     *
     * @param flag  Whether to track gradients.
     * @return This tensor, so calls can be chained.
     * @throws std::invalid_argument if this tensor is not a leaf.
     */
    Tensor& requires_grad_(bool flag = true);

    /**
     * @return Whether this tensor was not produced by a recorded operation.
     *         Parameters are leaves; intermediate results are not.
     */
    bool is_leaf() const;

    /**
     * The gradient accumulated for this tensor by backward().
     *
     * Only leaves that require a gradient ever get one.
     *
     * @return A pointer to the gradient, owned by this tensor, or nullptr if
     *         none has been accumulated.
     */
    const Tensor* grad() const;

    /**
     * Computes gradients back through the graph that produced this tensor.
     *
     * Starts from a gradient of 1 for this tensor and works backwards, adding
     * into the grad of every leaf that requires one. Gradients accumulate, so
     * calling this twice doubles them unless the gradients are cleared in
     * between.
     *
     * @throws std::invalid_argument if this tensor has any axes, since a
     *         starting gradient is only obvious for a scalar, or if it does
     *         not require a gradient.
     */
    void backward();

    /**
     * Returns this tensor's data without its autograd history.
     *
     * @return A tensor sharing this one's storage, requiring no gradient and
     *         holding no history, so gradients do not flow through it. Writes
     *         through either tensor are visible in the other.
     */
    Tensor detach() const;

    /**
     * This tensor's autograd state, shared with every copy of it.
     *
     * For the autograd layer: ops read it to decide whether to record, and
     * write the result's grad_fn through it.
     */
    std::shared_ptr<AutogradMeta> autograd_meta() const;
};

#endif
