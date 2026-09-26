#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <vector>
#include <memory>
#include <string>
#include <optional>

#include "slodl/tensor/tensor.hpp"

struct Edge {
    std::shared_ptr<Node> node;
    std::size_t input_nr = 0;
    bool is_valid() const;
};

class Node {
    public:
        std::vector<std::optional<Tensor>> apply(std::vector<std::optional<Tensor>> grad_out);
        std::vector<Edge> next_edges;
        std::string name;
        std::size_t num_outputs = 1;
        virtual ~Node() = default;
        Node& operator=(const Node&) = delete;
        Node(const Node&) = delete;
        Node() = default;
    
    protected:
        virtual std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) = 0;
        std::vector<std::vector<std::size_t>> input_shapes;
};

struct AutogradMeta {
    bool requires_grad = false;
    std::shared_ptr<Node> grad_fn;    // null for leaves
    std::size_t output_nr = 0;        // which output of grad_fn (only needed for multi-output ops)
    std::shared_ptr<Tensor> grad;     // filled for leaves by AccumulateGrad
};

#endif