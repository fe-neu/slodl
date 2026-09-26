#include <stdexcept>
#include <string>

#include "slodl/autograd/autograd.hpp"

bool Edge::is_valid() const {
    return node != nullptr;
}

namespace {

std::string format_shape(const std::vector<std::size_t>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); i++) {
        out += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return out + "]";
}

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
