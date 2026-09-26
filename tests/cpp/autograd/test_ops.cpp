#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "slodl/autograd/functions.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/autograd/ops.hpp"
#include "slodl/tensor/tensor.hpp"

TEST_CASE("add computes the sum", "[autograd]") {
    Tensor sum = add(Tensor({2}, {1.0, 2.0}), Tensor({2}, {10.0, 20.0}));

    CHECK(sum[0].item() == 11.0);
    CHECK(sum[1].item() == 22.0);
}

TEST_CASE("add records nothing when no input requires a gradient",
          "[autograd]") {
    Tensor sum = add(Tensor({2}, 1.0), Tensor({2}, 2.0));

    CHECK_FALSE(sum.requires_grad());
    CHECK(sum.is_leaf());
}

TEST_CASE("add records a graph when an input requires a gradient",
          "[autograd]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    a.requires_grad_();

    Tensor sum = add(a, b);

    CHECK(sum.requires_grad());
    CHECK_FALSE(sum.is_leaf());

    const std::shared_ptr<Node> node = sum.autograd_meta()->grad_fn;
    REQUIRE(node != nullptr);
    CHECK(node->name == "AddBackward");
    CHECK(sum.autograd_meta()->output_nr == 0);

    REQUIRE(node->next_edges.size() == 2);
    CHECK(node->next_edges[0].is_valid());
    CHECK_FALSE(node->next_edges[1].is_valid());
}

TEST_CASE("add points its edges at both inputs' accumulators", "[autograd]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    const std::shared_ptr<Node> node = add(a, b).autograd_meta()->grad_fn;

    REQUIRE(node->next_edges.size() == 2);
    CHECK(dynamic_cast<AccumulateGrad*>(node->next_edges[0].node.get()));
    CHECK(dynamic_cast<AccumulateGrad*>(node->next_edges[1].node.get()));
    CHECK(node->next_edges[0].node != node->next_edges[1].node);
}

TEST_CASE("add chains onto the node of a recorded input", "[autograd]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    Tensor c({2}, 3.0);
    a.requires_grad_();

    Tensor first = add(a, b);
    Tensor second = add(first, c);

    const std::shared_ptr<Node> node = second.autograd_meta()->grad_fn;
    REQUIRE(node->next_edges.size() == 2);
    CHECK(node->next_edges[0].node == first.autograd_meta()->grad_fn);
    CHECK(node->next_edges[0].input_nr == 0);
    CHECK_FALSE(node->next_edges[1].is_valid());
}

TEST_CASE("add records nothing inside a NoGradGuard", "[autograd]") {
    Tensor a({2}, 1.0);
    a.requires_grad_();

    NoGradGuard guard;
    Tensor sum = add(a, Tensor({2}, 2.0));

    CHECK(sum[0].item() == 3.0);
    CHECK_FALSE(sum.requires_grad());
    CHECK(sum.is_leaf());
}

TEST_CASE("add rejects mismatched shapes", "[autograd]") {
    CHECK_THROWS_AS(add(Tensor({2}), Tensor({3})), std::invalid_argument);
}

TEST_CASE("operator+ records like add", "[autograd]") {
    Tensor a({2}, 1.0);
    a.requires_grad_();

    Tensor sum = a + Tensor({2}, 2.0);

    CHECK(sum[0].item() == 3.0);
    CHECK(sum.autograd_meta()->grad_fn->name == "AddBackward");
}

TEST_CASE("AddBackward passes the gradient to both inputs", "[autograd]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    const std::shared_ptr<Node> node = add(a, b).autograd_meta()->grad_fn;
    std::vector<std::optional<Tensor>> gradients =
        node->apply({Tensor({2}, {3.0, 4.0})});

    REQUIRE(gradients.size() == 2);
    REQUIRE(gradients[0].has_value());
    REQUIRE(gradients[1].has_value());
    CHECK((*gradients[0])[0].item() == 3.0);
    CHECK((*gradients[0])[1].item() == 4.0);
    CHECK((*gradients[1])[0].item() == 3.0);
    CHECK((*gradients[1])[1].item() == 4.0);
}

TEST_CASE("a discarded graph frees its nodes", "[autograd]") {
    Tensor a({2}, 1.0);
    a.requires_grad_();

    {
        Tensor sum = add(a, Tensor({2}, 2.0));
        CHECK_FALSE(a.autograd_meta()->grad_accumulator.expired());
    }

    CHECK(a.autograd_meta()->grad_accumulator.expired());
}
