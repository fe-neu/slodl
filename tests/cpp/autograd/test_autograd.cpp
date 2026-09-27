#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "slodl/autograd/autograd.hpp"
#include "slodl/tensor/tensor.hpp"

namespace {

// A node whose backward returns whatever the test put in `gradients`, so each
// case can drive apply() into one specific outcome. `ran` records whether
// backward was reached at all.
struct ScriptedNode : Node {
    std::vector<std::optional<Tensor>> gradients;
    bool ran = false;

    ScriptedNode() { name = "ScriptedNode"; }

    void set_inputs(std::vector<std::vector<std::size_t>> shapes,
                    std::vector<Edge> edges) {
        input_shapes = std::move(shapes);
        next_edges = std::move(edges);
    }

    const std::vector<std::vector<std::size_t>>& shapes() const {
        return input_shapes;
    }

    std::vector<std::optional<Tensor>> backward(
        std::vector<std::optional<Tensor>> grad_out) override {
        (void)grad_out;
        ran = true;
        return gradients;
    }
};

Edge valid_edge() {
    return Edge{std::make_shared<ScriptedNode>(), 0};
}

std::vector<std::optional<Tensor>> one_gradient() {
    return {Tensor({2}, 1.0)};
}

}

TEST_CASE("an edge is valid only once it points at a node", "[autograd]") {
    CHECK_FALSE(Edge{}.is_valid());
    CHECK(valid_edge().is_valid());
}

TEST_CASE("apply passes the gradients from backward through", "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}}, {valid_edge()});
    node.gradients = {Tensor({2}, {3.0, 4.0})};

    std::vector<std::optional<Tensor>> results = node.apply(one_gradient());

    REQUIRE(node.ran);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].has_value());
    CHECK(results[0]->shape() == std::vector<std::size_t>{2});
    CHECK((*results[0])[0].item() == 3.0);
    CHECK((*results[0])[1].item() == 4.0);
}

TEST_CASE("apply accepts a missing gradient for an input that needs none",
          "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}, {3}}, {valid_edge(), Edge{}});
    node.gradients = {Tensor({2}, 1.0), std::nullopt};

    std::vector<std::optional<Tensor>> results = node.apply(one_gradient());

    REQUIRE(results.size() == 2);
    CHECK(results[0].has_value());
    CHECK_FALSE(results[1].has_value());
}

TEST_CASE("apply rejects the wrong number of incoming gradients",
          "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}}, {valid_edge()});
    node.gradients = one_gradient();

    CHECK_THROWS_AS(node.apply({}), std::invalid_argument);
    CHECK_THROWS_AS(node.apply({Tensor({2}, 1.0), Tensor({2}, 1.0)}),
                    std::invalid_argument);
    CHECK_FALSE(node.ran);
}

TEST_CASE("apply rejects a node whose input shapes and edges disagree",
          "[autograd]") {
    ScriptedNode node;
    node.set_inputs({}, {valid_edge()});
    node.gradients = one_gradient();

    CHECK_THROWS_AS(node.apply(one_gradient()), std::invalid_argument);
    CHECK_FALSE(node.ran);
}

TEST_CASE("apply rejects the wrong number of returned gradients",
          "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}, {2}}, {valid_edge(), valid_edge()});
    node.gradients = one_gradient();

    CHECK_THROWS_AS(node.apply(one_gradient()), std::invalid_argument);
}

TEST_CASE("apply rejects a missing gradient an input requires", "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}}, {valid_edge()});
    node.gradients = {std::nullopt};

    CHECK_THROWS_AS(node.apply(one_gradient()), std::invalid_argument);
}

TEST_CASE("apply rejects a gradient of the wrong shape", "[autograd]") {
    ScriptedNode node;
    node.set_inputs({{2}}, {valid_edge()});
    node.gradients = {Tensor({3}, 1.0)};

    CHECK_THROWS_AS(node.apply(one_gradient()), std::invalid_argument);
}

TEST_CASE("requires_grad_ is rejected on a non-leaf tensor", "[autograd]") {
    Tensor t({2});
    t.autograd_meta()->grad_fn = std::make_shared<ScriptedNode>();

    REQUIRE_FALSE(t.is_leaf());
    CHECK_THROWS_AS(t.requires_grad_(), std::invalid_argument);
}

TEST_CASE("AccumulateGrad writes the first gradient to the leaf", "[autograd]") {
    Tensor leaf({2}, {5.0, 6.0});
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    accumulator.apply({Tensor({2}, {1.0, 2.0})});

    REQUIRE(leaf.grad() != nullptr);
    CHECK(leaf.grad()->shape() == std::vector<std::size_t>{2});
    CHECK((*leaf.grad())[0].item() == 1.0);
    CHECK((*leaf.grad())[1].item() == 2.0);
}

TEST_CASE("AccumulateGrad sums repeated gradients", "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    accumulator.apply({Tensor({2}, {1.0, 2.0})});
    accumulator.apply({Tensor({2}, {10.0, 20.0})});
    accumulator.apply({Tensor({2}, {100.0, 200.0})});

    CHECK((*leaf.grad())[0].item() == 111.0);
    CHECK((*leaf.grad())[1].item() == 222.0);
}

TEST_CASE("AccumulateGrad returns no gradients and has no edges", "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    CHECK(accumulator.next_edges.empty());
    CHECK(accumulator.apply({Tensor({2}, 1.0)}).empty());
}

TEST_CASE("the accumulated gradient is independent of the incoming tensor",
          "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    Tensor incoming({2}, {1.0, 2.0});
    accumulator.apply({incoming});
    incoming[0].fill_(99.0);

    CHECK((*leaf.grad())[0].item() == 1.0);
}

TEST_CASE("AccumulateGrad accumulates a non-contiguous gradient", "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    Tensor matrix({2, 2}, {1.0, 2.0, 3.0, 4.0});
    accumulator.apply({matrix[1]});

    CHECK((*leaf.grad())[0].item() == 3.0);
    CHECK((*leaf.grad())[1].item() == 4.0);
}

TEST_CASE("AccumulateGrad rejects a gradient of the wrong shape", "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();
    AccumulateGrad accumulator(leaf);

    CHECK_THROWS_AS(accumulator.apply({Tensor({3}, 1.0)}),
                    std::invalid_argument);
    CHECK_THROWS_AS(accumulator.apply({std::nullopt}), std::invalid_argument);
    CHECK(leaf.grad() == nullptr);
}

TEST_CASE("AccumulateGrad rejects a non-leaf tensor", "[autograd]") {
    Tensor t({2});
    t.autograd_meta()->grad_fn = std::make_shared<ScriptedNode>();

    CHECK_THROWS_AS(AccumulateGrad(t), std::invalid_argument);
}

TEST_CASE("gradient_edge points a leaf at its accumulator", "[autograd]") {
    Tensor leaf({2});
    leaf.requires_grad_();

    Edge edge = gradient_edge(leaf);

    REQUIRE(edge.is_valid());
    CHECK(edge.input_nr == 0);
    CHECK(dynamic_cast<AccumulateGrad*>(edge.node.get()) != nullptr);
}

TEST_CASE("gradient_edge reuses one accumulator per leaf", "[autograd]") {
    Tensor leaf({2});
    leaf.requires_grad_();

    Edge first = gradient_edge(leaf);
    Edge second = gradient_edge(leaf);

    CHECK(first.node == second.node);
}

TEST_CASE("an accumulator is owned by the graph, not the leaf", "[autograd]") {
    Tensor leaf({2});
    leaf.requires_grad_();

    {
        Edge edge = gradient_edge(leaf);
        CHECK_FALSE(leaf.autograd_meta()->grad_accumulator.expired());
    }

    CHECK(leaf.autograd_meta()->grad_accumulator.expired());
}

TEST_CASE("gradient_edge is invalid for a tensor that needs no gradient",
          "[autograd]") {
    Tensor plain({2});

    CHECK_FALSE(gradient_edge(plain).is_valid());
}

TEST_CASE("gradient_edge points a non-leaf at its grad_fn", "[autograd]") {
    std::shared_ptr<ScriptedNode> producer = std::make_shared<ScriptedNode>();
    Tensor result({2});
    result.autograd_meta()->requires_grad = true;
    result.autograd_meta()->grad_fn = producer;
    result.autograd_meta()->output_nr = 2;

    Edge edge = gradient_edge(result);

    CHECK(edge.node == producer);
    CHECK(edge.input_nr == 2);
}

TEST_CASE("collect_inputs fills edges and shapes in input order",
          "[autograd]") {
    Tensor tracked({2, 2});
    tracked.requires_grad_();
    Tensor plain({3});

    ScriptedNode node;
    node.collect_inputs({tracked, plain});

    REQUIRE(node.next_edges.size() == 2);
    CHECK(node.next_edges[0].is_valid());
    CHECK_FALSE(node.next_edges[1].is_valid());

    REQUIRE(node.shapes().size() == 2);
    CHECK(node.shapes()[0] == std::vector<std::size_t>{2, 2});
    CHECK(node.shapes()[1] == std::vector<std::size_t>{3});
}

TEST_CASE("collect_inputs replaces what an earlier call wired up",
          "[autograd]") {
    Tensor leaf({2});
    leaf.requires_grad_();

    ScriptedNode node;
    node.collect_inputs({leaf, leaf});
    node.collect_inputs({leaf});

    CHECK(node.next_edges.size() == 1);
    CHECK(node.shapes().size() == 1);
}

TEST_CASE("using one leaf twice reaches a single accumulator", "[autograd]") {
    Tensor leaf({2}, 0.0);
    leaf.requires_grad_();

    ScriptedNode node;
    node.collect_inputs({leaf, leaf});

    REQUIRE(node.next_edges[0].node == node.next_edges[1].node);

    node.next_edges[0].node->apply({Tensor({2}, {1.0, 2.0})});
    node.next_edges[1].node->apply({Tensor({2}, {10.0, 20.0})});

    CHECK((*leaf.grad())[0].item() == 11.0);
    CHECK((*leaf.grad())[1].item() == 22.0);
}
