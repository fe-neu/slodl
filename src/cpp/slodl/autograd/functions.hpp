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

#endif
