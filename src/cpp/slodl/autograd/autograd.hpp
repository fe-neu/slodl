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

        /**
         * Wires this node to the tensors its operation consumed.
         *
         * Fills next_edges and input_shapes in one pass, so the two stay
         * aligned: entry i of both describes input i of the forward
         * operation. Any previous contents are replaced.
         *
         * @param inputs  The operation's inputs, in the order it received
         *                them, which is the order backward() must return
         *                gradients in.
         */
        void collect_inputs(const std::vector<Tensor>& inputs);
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

    std::weak_ptr<Node> grad_accumulator;
};

/**
 * The edge along which a tensor's gradient should travel.
 *
 * @param tensor  An input of an operation being recorded.
 * @return An edge to the tensor's grad_fn if it has one; to its
 *         AccumulateGrad, creating and caching it on first use, if it is a
 *         leaf that requires a gradient; an invalid edge otherwise, meaning
 *         the gradient for this input is discarded.
 */
Edge gradient_edge(const Tensor& tensor);

class AccumulateGrad : public Node {
    public:
        explicit AccumulateGrad(const Tensor& leaf);

    protected:
        std::vector<std::optional<Tensor>> backward(std::vector<std::optional<Tensor>> grad_out) override;

    private:
        std::shared_ptr<AutogradMeta> leaf_meta;
        std::vector<std::size_t> leaf_shape;
};

#endif