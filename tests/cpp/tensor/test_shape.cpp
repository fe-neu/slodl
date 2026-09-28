#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "slodl/tensor/shape.hpp"

using Shape = std::vector<std::size_t>;

TEST_CASE("broadcast_shapes takes the larger of each dimension", "[tensor]") {
    CHECK(broadcast_shapes({2, 3}, {2, 3}) == Shape{2, 3});
    CHECK(broadcast_shapes({2, 3}, {3}) == Shape{2, 3});
    CHECK(broadcast_shapes({4, 1, 3}, {5, 3}) == Shape{4, 5, 3});
    CHECK(broadcast_shapes({2, 1}, {1, 3}) == Shape{2, 3});
}

TEST_CASE("broadcast_shapes treats a scalar as compatible with anything",
          "[tensor]") {
    CHECK(broadcast_shapes({}, {2, 3}) == Shape{2, 3});
    CHECK(broadcast_shapes({2, 3}, {}) == Shape{2, 3});
    CHECK(broadcast_shapes({}, {}) == Shape{});
}

TEST_CASE("broadcast_shapes rejects incompatible dimensions", "[tensor]") {
    CHECK_THROWS_AS(broadcast_shapes({2}, {3}), std::invalid_argument);
    CHECK_THROWS_AS(broadcast_shapes({2, 3}, {2, 4}), std::invalid_argument);
}

TEST_CASE("broadcast_strides zeroes the stretched axes", "[tensor]") {
    // A [3] read as [2, 3]: the new leading axis never moves.
    CHECK(broadcast_strides({3}, {1}, {2, 3}) == Shape{0, 1});

    // A [2, 1] read as [2, 3]: the stretched axis never moves.
    CHECK(broadcast_strides({2, 1}, {1, 1}, {2, 3}) == Shape{1, 0});

    // Already the right shape: strides are kept.
    CHECK(broadcast_strides({2, 3}, {3, 1}, {2, 3}) == Shape{3, 1});

    // A scalar read as anything: nothing moves.
    CHECK(broadcast_strides({}, {}, {2, 2}) == Shape{0, 0});
}

TEST_CASE("broadcast_strides rejects shapes it cannot stretch", "[tensor]") {
    CHECK_THROWS_AS(broadcast_strides({2}, {1}, {3}), std::invalid_argument);
    CHECK_THROWS_AS(broadcast_strides({2, 3}, {3, 1}, {3}), std::invalid_argument);
}
