#ifndef AUTOGRAD_OPS_HPP
#define AUTOGRAD_OPS_HPP

#include <memory>
#include <vector>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/tensor.hpp"

/**
 * Whether an operation on these inputs should be recorded for autograd.
 *
 * @param inputs  The operation's inputs.
 * @return True if recording is enabled and at least one input requires a
 *         gradient. Every op asks this one question, so the rule lives here
 *         rather than in each of them.
 */
bool should_record(const std::vector<Tensor>& inputs);

/**
 * Makes `result` the recorded output of `node`, and wires `node` to `inputs`.
 *
 * @param result  The operation's freshly computed output, which must carry no
 *                history of its own yet.
 * @param node    The backward node for this operation.
 * @param inputs  The operation's inputs, in the order it received them.
 */
void set_history(
    Tensor& result,
    std::shared_ptr<Node> node,
    const std::vector<Tensor>& inputs
);

/**
 * Adds two tensors, recording the operation for autograd.
 *
 * @param a  Left operand.
 * @param b  Right operand, which must have exactly the shape of `a`.
 * @return The sum. It requires a gradient, and carries an AddBackward as its
 *         grad_fn, if either input requires a gradient and recording is
 *         enabled; otherwise it is a plain leaf.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor add(const Tensor& a, const Tensor& b);

/** Adds two tensors; see add(). */
Tensor operator+(const Tensor& a, const Tensor& b);

/**
 * calude do this
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor sub(const Tensor& a, const Tensor& b);

/** Subtract two tensors; see sub(). */
Tensor operator-(const Tensor& a, const Tensor& b);

/**
 * calude do this
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor neg(const Tensor& a);

/** claude */
Tensor operator-(const Tensor& a);

/**
 * Calculates the Hadamard product of two tensors, recording the operation for autograd.
 *
 * @param a  Left operand.
 * @param b  Right operand, which must have exactly the shape of `a`.
 * @return The Hadamard product. It requires a gradient, and carries a MulBackward as its
 *         grad_fn, if either input requires a gradient and recording is
 *         enabled; otherwise it is a plain leaf.
 * @throws std::invalid_argument if the two shapes differ.
 */
Tensor mul(const Tensor& a, const Tensor& b);

/** Calculates Hadamard product of two tensors; see mul(). */
Tensor operator*(const Tensor& a, const Tensor& b);

/**
 * Adds up every element of a tensor, recording the operation for autograd.
 *
 * This is how a tensor becomes the scalar that backward() can start from.
 *
 * @param a  Tensor to add up, which may be a view.
 * @return A 0-dimensional tensor holding the total. It requires a gradient,
 *         and carries a SumBackward as its grad_fn, if the input requires a
 *         gradient and recording is enabled; otherwise it is a plain leaf.
 */
Tensor sum(const Tensor& a);

#endif
