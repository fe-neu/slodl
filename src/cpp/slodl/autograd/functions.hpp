#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <optional>
#include <vector>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/tensor.hpp"

/**
 * Backward of addition.
 *
 * Adding does not scale its inputs, so both partial derivatives are 1 and the
 * incoming gradient passes through unchanged to each input. Nothing from the
 * forward pass is needed, which makes this node stateless.
 */
class AddBackward : public Node {
    public:
        AddBackward();

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;
};

/**
 * Backward of subtraction.
 *
 * Raising the left operand raises the difference by the same amount, while
 * raising the right operand lowers it, so the partial derivatives are 1 and
 * -1. The left input therefore receives the incoming gradient unchanged and
 * the right one receives it negated.
 *
 * Nothing from the forward pass is needed, which makes this node stateless.
 */
class SubBackward : public Node {
    public:
        SubBackward();

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;
};

/**
 * Backward of negation.
 *
 * Negating scales its input by -1, so the partial derivative is -1 and the
 * incoming gradient comes back negated.
 *
 * Nothing from the forward pass is needed, which makes this node stateless.
 */
class NegBackward : public Node {
    public:
        NegBackward();

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;
};

/**
 * Backward of element-wise multiplication.
 *
 * Each input is scaled by the other, so the partial derivative with respect to
 * one input is the other input's value, and each gradient is the incoming
 * gradient multiplied by the opposite input.
 *
 * That makes this node stateful: it has to remember both inputs from the
 * forward pass. They are stored detached, so the saved values carry no history
 * of their own and cannot keep the graph that produced them alive.
 */
class MulBackward : public Node {
    public:
        MulBackward(const Tensor& a, const Tensor& b);

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;

    private:
        // By value, not by reference: these are detached copies of the inputs,
        // and a reference would dangle once the constructor's temporaries die.
        // Copying a Tensor shares its storage, so this is cheap.
        Tensor a;
        Tensor b;
};

/**
 * Backward of summing every element.
 *
 * A sum is linear in each element: nudging one element moves the total by the
 * same amount, so every partial derivative is 1 and each element's gradient is
 * the incoming gradient itself. Since the sum is a scalar, that gradient is a
 * single number, copied into every position of a tensor shaped like the input.
 *
 * A reduction forward is an expansion backward, which is the mirror of
 * broadcasting, where a copy forward becomes a sum backward.
 *
 * Nothing from the forward pass is needed beyond the input's shape, which
 * Node already records in input_shapes, so this node is stateless.
 */
class SumBackward : public Node {
    public:
        SumBackward();

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;
};
#endif
