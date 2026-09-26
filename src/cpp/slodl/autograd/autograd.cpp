#include <stdexcept>
#include <string>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/ops.hpp"

bool Edge::is_valid() const {
    return node != nullptr;
}

std::vector<std::optional<Tensor>> Node::apply(std::vector<std::optional<Tensor>> grad_out) {
    if (grad_out.size() != num_outputs) {
        throw std::invalid_argument(
            name + ": got " + std::to_string(grad_out.size()) +
            " incoming gradients, expected " + std::to_string(num_outputs));
    }
    if (input_shapes.size() != next_edges.size()) {
        throw std::invalid_argument(
            name + ": has " + std::to_string(input_shapes.size()) +
            " input shapes but " + std::to_string(next_edges.size()) +
            " next edges");
    }

    std::vector<std::optional<Tensor>> results = backward(std::move(grad_out));

    if (next_edges.size() != results.size()) {
        throw std::invalid_argument(
            name + ": backward returned " + std::to_string(results.size()) +
            " gradients, expected " + std::to_string(next_edges.size()));
    }

    for(std::size_t i = 0; i < results.size(); i++) {
        if (!next_edges[i].is_valid()) {
            continue;
        }
        if (!results[i].has_value()) {
            throw std::invalid_argument(
                name + ": backward returned no gradient for input " +
                std::to_string(i) + ", which requires one");
        }
        if (results[i]->shape() != input_shapes[i]) {
            throw std::invalid_argument(
                name + ": backward returned a gradient of shape " +
                format_shape(results[i]->shape()) + " for input " +
                std::to_string(i) + ", expected " +
                format_shape(input_shapes[i]));
        }
    }
    return results;
}

AccumulateGrad::AccumulateGrad(const Tensor& leaf)
    : leaf_meta(leaf.autograd_meta()),
    leaf_shape(leaf.shape()) {
        name = "AccumulateGrad";
        if (!leaf.is_leaf()) {
            throw std::invalid_argument(
                "AccumulateGrad: the tensor is not a leaf");
        }
    }

std::vector<std::optional<Tensor>> AccumulateGrad::backward(
    std::vector<std::optional<Tensor>> grad_out
) {
    if (!grad_out[0].has_value()) {
        throw std::invalid_argument(
            "AccumulateGrad: got no gradient to accumulate");
    }
    if (grad_out[0]->shape() != leaf_shape) {
        throw std::invalid_argument(
            "AccumulateGrad: got a gradient of shape " +
            format_shape(grad_out[0]->shape()) + ", expected " +
            format_shape(leaf_shape));
    }

    Tensor incoming = grad_out[0]->clone();

    if (!leaf_meta->grad) {
        leaf_meta->grad = std::make_shared<Tensor>(std::move(incoming));
        return {};
    }

    double* accumulated = leaf_meta->grad->data();
    const double* addend = incoming.data();
    const std::size_t count = element_count(leaf_shape);
    for (std::size_t i = 0; i < count; i++) {
        accumulated[i] += addend[i];
    }
    return {};
}
