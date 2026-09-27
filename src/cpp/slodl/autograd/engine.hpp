#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <memory>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/tensor.hpp"

/**
 * Runs a backward pass through the graph reachable from one node.
 *
 * Walks the graph backwards from `root`, calling each node exactly once, and
 * leaves the result in the `grad` of every leaf that requires one. A node runs
 * only after every gradient flowing into it has arrived, and gradients that
 * meet at the same node are summed, so a tensor used more than once
 * contributes each of its uses.
 *
 * Nothing is recorded while this runs, so no gradient carries a history of its
 * own and computing gradients of gradients is not possible.
 *
 * @param root  The node to start from, usually the grad_fn of the tensor
 *              backward() was called on.
 * @param seed  The gradient of that tensor with respect to itself, which must
 *              have the shape of the tensor `root` produced.
 * @throws std::invalid_argument if a node reports gradients that do not match
 *         its inputs; see Node::apply.
 */
void run_backward(const std::shared_ptr<Node>& root, const Tensor& seed);

#endif
