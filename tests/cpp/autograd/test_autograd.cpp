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
