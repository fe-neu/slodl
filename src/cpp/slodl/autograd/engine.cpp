#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "slodl/autograd/engine.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/tensor/ops.hpp"

namespace {

// Nodes are keyed by address. The graph owns them through the root's chain of
// shared_ptrs, so every pointer stays valid for the whole pass.
using DependencyCounts = std::unordered_map<const Node*, std::size_t>;
using GradientBuffers =
    std::unordered_map<const Node*, std::vector<std::optional<Tensor>>>;

/**
 * Counts, per node, how many gradients it should expect.
 *
 * A node may be reached along several edges, once per use of the tensor it
 * produced, and it must not run until all of them have delivered. Walking the
 * graph once up front is what makes that decidable: without the counts there
 * is no way to tell a node that is merely unfinished from one that is ready.
 */
DependencyCounts count_dependencies(const std::shared_ptr<Node>& root) {
    DependencyCounts counts;

    // An explicit stack rather than recursion: a deep graph would otherwise
    // grow the call stack with it.
    std::vector<const Node*> pending = {root.get()};
    std::unordered_map<const Node*, bool> seen = {{root.get(), true}};

    while (!pending.empty()) {
        const Node* node = pending.back();
        pending.pop_back();

        for (const Edge& edge : node->next_edges) {
            if (!edge.is_valid()) {
                continue;
            }
            const Node* target = edge.node.get();
            counts[target]++;
            if (!seen[target]) {
                seen[target] = true;
                pending.push_back(target);
            }
        }
    }
    return counts;
}

/** Sums a gradient into one slot of a node's buffer. */
void accumulate(
    std::vector<std::optional<Tensor>>& buffer,
    std::size_t slot,
    const Tensor& gradient
) {
    if (slot >= buffer.size()) {
        throw std::invalid_argument(
            "backward: an edge points at input " + std::to_string(slot) +
            " of a node that has only " + std::to_string(buffer.size()));
    }
    if (!buffer[slot]) {
        buffer[slot] = gradient;
        return;
    }
    // emplace, not assignment: assigning to an optional that already holds a
    // tensor would run Tensor::operator=, which writes values into that
    // tensor's storage. Gradients are shared, so that would overwrite what
    // other nodes still have to read. emplace replaces the tensor instead.
    //
    // add_kernel, not add: the backward pass records nothing.
    buffer[slot].emplace(add_kernel(*buffer[slot], gradient));
}

}

void run_backward(const std::shared_ptr<Node>& root, const Tensor& seed) {
    if (!root) {
        throw std::invalid_argument("backward: there is no graph to run");
    }

    // Belt and braces: the formulas below call kernels, which never record,
    // but this states the guarantee instead of relying on it.
    NoGradGuard guard;

    DependencyCounts remaining = count_dependencies(root);

    GradientBuffers buffers;
    buffers[root.get()] =
        std::vector<std::optional<Tensor>>(root->num_outputs);
    accumulate(buffers[root.get()], 0, seed);

    // Nodes whose gradients have all arrived. The root qualifies immediately:
    // the seed is the only gradient flowing into it.
    std::vector<std::shared_ptr<Node>> ready = {root};

    while (!ready.empty()) {
        const std::shared_ptr<Node> node = std::move(ready.back());
        ready.pop_back();

        std::vector<std::optional<Tensor>> incoming =
            std::move(buffers[node.get()]);
        buffers.erase(node.get());

        const std::vector<std::optional<Tensor>> gradients =
            node->apply(std::move(incoming));

        for (std::size_t i = 0; i < gradients.size(); i++) {
            const Edge& edge = node->next_edges[i];
            if (!edge.is_valid() || !gradients[i]) {
                continue;
            }

            const Node* target = edge.node.get();
            if (buffers.find(target) == buffers.end()) {
                buffers[target] =
                    std::vector<std::optional<Tensor>>(edge.node->num_outputs);
            }
            accumulate(buffers[target], edge.input_nr, *gradients[i]);

            // Only once every gradient has arrived is the target's buffer
            // complete, and only then may it run.
            if (--remaining[target] == 0) {
                ready.push_back(edge.node);
            }
        }
    }
}
