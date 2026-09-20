#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "slodl/tensor/tensor_storage.hpp"

TEST_CASE("TensorStorage(size, init_value) fills the buffer", "[tensor][tensor_storage]") {
    TensorStorage storage(3, 7.0);

    REQUIRE(storage.size() == 3);
    for (std::size_t i = 0; i < storage.size(); ++i) {
        CHECK(storage.ptr()[i] == 7.0);
    }
}

TEST_CASE("TensorStorage(data) takes over the given elements", "[tensor][tensor_storage]") {
    TensorStorage storage(std::vector<double>{1.0, 2.0, 3.0});

    REQUIRE(storage.size() == 3);
    CHECK(storage.ptr()[0] == 1.0);
    CHECK(storage.ptr()[2] == 3.0);
}

TEST_CASE("ptr() hands out a writable buffer", "[tensor][tensor_storage]") {
    TensorStorage storage(2, 0.0);

    storage.ptr()[1] = 42.0;

    CHECK(storage.ptr()[1] == 42.0);
    CHECK(storage.ptr()[0] == 0.0);
}

TEST_CASE("const ptr() reads the same buffer", "[tensor][tensor_storage]") {
    const TensorStorage storage(std::vector<double>{1.0, 2.0});

    CHECK(storage.ptr()[0] == 1.0);
    CHECK(storage.ptr()[1] == 2.0);
}

TEST_CASE("an empty TensorStorage is allowed", "[tensor][tensor_storage]") {
    TensorStorage storage(0, 0.0);

    CHECK(storage.size() == 0);
}
