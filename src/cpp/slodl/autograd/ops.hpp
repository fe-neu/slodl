#ifndef AUTOGRAD_OPS_HPP
#define AUTOGRAD_OPS_HPP

#include <memory>
#include <vector>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/tensor.hpp"

/**
 * Reads a tensor as though it had a larger shape, recording the operation.
 *
 * Nothing is copied; the result is a view with a stride of 0 along every
 * stretched axis. In a backward pass the gradient is summed back down to the
 * original shape, because one element was read in several places.
 *
 * @param a      Tensor to stretch.
 * @param shape  Shape to read it as, which `a`'s shape must broadcast to.
 * @return The stretched view. It requires a gradient, and carries an
 *         ExpandBackward as its grad_fn, if `a` requires a gradient and
 *         recording is enabled.
 * @throws std::invalid_argument if `a`'s shape does not broadcast to `shape`.
 */
Tensor expand(const Tensor& a, const std::vector<std::size_t>& shape);

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
 * @param b  Right operand, whose shape must broadcast against `a`'s.
 * @return The sum, of the two shapes broadcast together. It requires a gradient, and carries an AddBackward as its
 *         grad_fn, if either input requires a gradient and recording is
 *         enabled; otherwise it is a plain leaf.
 * @throws std::invalid_argument if the two shapes cannot be broadcast
 *         together.
 */
Tensor add(const Tensor& a, const Tensor& b);

/** Adds two tensors; see add(). */
Tensor operator+(const Tensor& a, const Tensor& b);

/**
 * Subtracts one tensor from another, recording the operation for autograd.
 *
 * @param a  Left operand, the tensor to subtract from.
 * @param b  Right operand, subtracted from `a`, whose shape must broadcast
 *           against `a`'s.
 * @return The difference, of the two shapes broadcast together. It requires a
 *         gradient, and carries a SubBackward as its grad_fn, if either input
 *         requires a gradient and recording is enabled; otherwise it is a
 *         plain leaf.
 * @throws std::invalid_argument if the two shapes cannot be broadcast
 *         together.
 */
Tensor sub(const Tensor& a, const Tensor& b);

/** Subtract two tensors; see sub(). */
Tensor operator-(const Tensor& a, const Tensor& b);

/**
 * Flips the sign of every element, recording the operation for autograd.
 *
 * @param a  Tensor to negate, which may be a view.
 * @return The negated tensor. It requires a gradient, and carries a
 *         NegBackward as its grad_fn, if `a` requires a gradient and recording
 *         is enabled; otherwise it is a plain leaf.
 */
Tensor neg(const Tensor& a);

/** Negates a tensor; see neg(). */
Tensor operator-(const Tensor& a);

/**
 * Calculates the Hadamard product of two tensors, recording the operation for autograd.
 *
 * @param a  Left operand.
 * @param b  Right operand, whose shape must broadcast against `a`'s.
 * @return The Hadamard product, of the two shapes broadcast together. It requires a gradient, and carries a MulBackward as its
 *         grad_fn, if either input requires a gradient and recording is
 *         enabled; otherwise it is a plain leaf.
 * @throws std::invalid_argument if the two shapes cannot be broadcast
 *         together.
 */
Tensor mul(const Tensor& a, const Tensor& b);

/** Calculates Hadamard product of two tensors; see mul(). */
Tensor operator*(const Tensor& a, const Tensor& b);

/**
 * Divides two tensors element by element, recording the operation.
 *
 * @param a  Left operand, the dividend.
 * @param b  Right operand, the divisor, whose shape must broadcast against
 *           `a`'s. Dividing by zero yields an infinity or a NaN rather than
 *           throwing.
 * @return The quotient. It requires a gradient, and carries a DivBackward as
 *         its grad_fn, if either input requires a gradient and recording is
 *         enabled; otherwise it is a plain leaf.
 * @throws std::invalid_argument if the two shapes cannot be broadcast
 *         together.
 */
Tensor div(const Tensor& a, const Tensor& b);

/** Divides two tensors; see div(). */
Tensor operator/(const Tensor& a, const Tensor& b);

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


/**
 * Averages every element of a tensor, recording the operation for autograd.
 *
 * Composed from sum() and div() rather than given a node of its own: the graph
 * those two record already differentiates correctly, giving every element a
 * gradient of 1/n. Its grad_fn is therefore a DivBackward, not a node named
 * after the mean.
 *
 * @param a  Tensor to average, which may be a view.
 * @return A 0-dimensional tensor holding the average, or NaN for an empty
 *         tensor, since that divides zero by zero.
 */
Tensor mean(const Tensor& a);

/**
 * Scalar forms of the element-wise operations.
 *
 * A plain number is treated as a 0-dimensional tensor and broadcast against
 * the other operand, so `t * 2.0` scales every element. The number requires no
 * gradient, so only the tensor's side of the graph is recorded.
 */
Tensor add(const Tensor& a, double b);
Tensor add(double a, const Tensor& b);
Tensor operator+(const Tensor& a, double b);
Tensor operator+(double a, const Tensor& b);
Tensor sub(const Tensor& a, double b);
Tensor sub(double a, const Tensor& b);
Tensor operator-(const Tensor& a, double b);
Tensor operator-(double a, const Tensor& b);
Tensor mul(const Tensor& a, double b);
Tensor mul(double a, const Tensor& b);
Tensor operator*(const Tensor& a, double b);
Tensor operator*(double a, const Tensor& b);
Tensor div(const Tensor& a, double b);
Tensor div(double a, const Tensor& b);
Tensor operator/(const Tensor& a, double b);
Tensor operator/(double a, const Tensor& b);

#endif
