#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

#include "slodl/autograd/autograd.hpp"
#include "slodl/autograd/engine.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/autograd/ops.hpp"
#include "slodl/tensor/tensor.hpp"

namespace {

Tensor scalar(double value) {
    return Tensor({}, value);
}

Tensor tracked_scalar(double value) {
    Tensor t = scalar(value);
    t.requires_grad_();
    return t;
}

}

TEST_CASE("backward gives each input of an addition a gradient of one",
          "[autograd]") {
    Tensor a = tracked_scalar(2.0);
    Tensor b = tracked_scalar(3.0);

    Tensor sum = add(a, b);
    sum.backward();

    REQUIRE(a.grad() != nullptr);
    REQUIRE(b.grad() != nullptr);
    CHECK(a.grad()->item() == 1.0);
    CHECK(b.grad()->item() == 1.0);
}

TEST_CASE("backward leaves inputs that require no gradient alone",
          "[autograd]") {
    Tensor a = tracked_scalar(2.0);
    Tensor b = scalar(3.0);

    add(a, b).backward();

    CHECK(a.grad()->item() == 1.0);
    CHECK(b.grad() == nullptr);
}

TEST_CASE("a tensor used twice collects both gradients", "[autograd]") {
    Tensor a = tracked_scalar(2.0);

    add(a, a).backward();

    CHECK(a.grad()->item() == 2.0);
}

TEST_CASE("backward follows a chain of operations", "[autograd]") {
    Tensor a = tracked_scalar(1.0);
    Tensor b = tracked_scalar(2.0);
    Tensor c = tracked_scalar(3.0);

    add(add(a, b), c).backward();

    CHECK(a.grad()->item() == 1.0);
    CHECK(b.grad()->item() == 1.0);
    CHECK(c.grad()->item() == 1.0);
}

TEST_CASE("a chain that reuses a tensor sums along both routes", "[autograd]") {
    Tensor a = tracked_scalar(1.0);
    Tensor b = tracked_scalar(2.0);

    // a appears twice: once through the inner sum, once directly.
    add(add(a, b), a).backward();

    CHECK(a.grad()->item() == 2.0);
    CHECK(b.grad()->item() == 1.0);
}

TEST_CASE("a node feeding two consumers runs once, after both", "[autograd]") {
    Tensor a = tracked_scalar(1.0);
    Tensor b = tracked_scalar(2.0);

    // shared is consumed by two additions, which meet again at the root, so
    // its node must wait for both before running.
    Tensor shared = add(a, b);
    Tensor left = add(shared, a);
    Tensor right = add(shared, b);
    add(left, right).backward();

    // a: two routes through shared, plus one direct. b likewise.
    CHECK(a.grad()->item() == 3.0);
    CHECK(b.grad()->item() == 3.0);
}

TEST_CASE("gradients accumulate across backward passes", "[autograd]") {
    Tensor a = tracked_scalar(2.0);
    Tensor b = tracked_scalar(3.0);

    add(a, b).backward();
    add(a, b).backward();

    CHECK(a.grad()->item() == 2.0);
}

TEST_CASE("backward on a tracked scalar leaf fills its own gradient",
          "[autograd]") {
    Tensor a = tracked_scalar(2.0);

    a.backward();

    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->item() == 1.0);
}

TEST_CASE("backward is rejected without a scalar that requires a gradient",
          "[autograd]") {
    Tensor vector_valued({2}, 1.0);
    vector_valued.requires_grad_();
    CHECK_THROWS_AS(add(vector_valued, vector_valued).backward(),
                    std::invalid_argument);

    Tensor untracked = scalar(1.0);
    CHECK_THROWS_AS(untracked.backward(), std::invalid_argument);
}

TEST_CASE("gradients of larger tensors keep their shape", "[autograd]") {
    Tensor a({2, 2}, 1.0);
    Tensor b({2, 2}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    Tensor sum = add(a, b);
    run_backward(sum.autograd_meta()->grad_fn, Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}));

    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->shape() == std::vector<std::size_t>{2, 2});
    CHECK((*a.grad())[0][0].item() == 1.0);
    CHECK((*a.grad())[1][1].item() == 4.0);
    CHECK((*b.grad())[1][0].item() == 3.0);
}

TEST_CASE("the backward pass records nothing", "[autograd]") {
    Tensor a = tracked_scalar(2.0);

    add(a, a).backward();

    CHECK(a.grad()->is_leaf());
    CHECK_FALSE(a.grad()->requires_grad());
    CHECK(is_grad_enabled());
}

TEST_CASE("run_backward rejects a missing graph", "[autograd]") {
    CHECK_THROWS_AS(run_backward(nullptr, scalar(1.0)), std::invalid_argument);
}
