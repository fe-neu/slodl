#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "slodl/tensor/tensor.hpp"

TEST_CASE("Tensor(dims) zero-fills every element", "[tensor]") {
    Tensor t({2, 3});

    CHECK(t.shape() == std::vector<std::size_t>{2, 3});
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(t[i][j].item() == 0.0);
        }
    }
}

TEST_CASE("Tensor(dims, init_value) fills every element", "[tensor]") {
    Tensor t({2, 2}, 7.0);

    CHECK(t[0][0].item() == 7.0);
    CHECK(t[1][1].item() == 7.0);
}

TEST_CASE("Tensor(dims, data) stores elements row-major", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});

    CHECK(t[0][0].item() == 1.0);
    CHECK(t[0][1].item() == 2.0);
    CHECK(t[1][0].item() == 3.0);
    CHECK(t[1][1].item() == 4.0);
}

TEST_CASE("Tensor(dims, data) rejects data of the wrong length", "[tensor]") {
    CHECK_THROWS_AS(Tensor({2, 2}, std::vector<double>{1.0, 2.0, 3.0}),
                    std::out_of_range);
    CHECK_THROWS_AS(Tensor({2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),
                    std::out_of_range);
}

TEST_CASE("strides are row-major and counted in elements", "[tensor]") {
    CHECK(Tensor({3, 3, 3}).element_strides() == std::vector<std::size_t>{9, 3, 1});
    CHECK(Tensor({2, 5}).element_strides() == std::vector<std::size_t>{5, 1});
    CHECK(Tensor({4}).element_strides() == std::vector<std::size_t>{1});
    CHECK(Tensor({}).element_strides().empty());
}

TEST_CASE("a 0-dimensional tensor holds a single element", "[tensor]") {
    Tensor scalar({}, 5.0);

    CHECK(scalar.shape().empty());
    CHECK(scalar.item() == 5.0);
}

TEST_CASE("operator[] returns a view onto the same storage", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor row = t[1];

    CHECK(row.shape() == std::vector<std::size_t>{2});
    CHECK(row.data() == t.data() + 2);  // points into the parent's buffer

    row[0] = 30.0;

    CHECK(t[1][0].item() == 30.0);  // the write is visible from the parent
}

TEST_CASE("indexing leaves the tensor it came from unchanged", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});

    (void)t[0];

    CHECK(t.shape() == std::vector<std::size_t>{2, 2});
    CHECK(t.element_strides() == std::vector<std::size_t>{2, 1});
}

TEST_CASE("indexing every dimension yields a 0-dimensional tensor", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});

    CHECK(t[1][1].shape().empty());
    CHECK(t[1][1].item() == 4.0);
}

TEST_CASE("operator[] rejects an out-of-range index", "[tensor]") {
    Tensor t({2, 2});

    CHECK_THROWS_AS(t[2], std::out_of_range);
    CHECK_THROWS_AS(t[0][2], std::out_of_range);
}

TEST_CASE("a 0-dimensional tensor cannot be indexed", "[tensor]") {
    Tensor scalar({}, 1.0);

    CHECK_THROWS_AS(scalar[0], std::out_of_range);
}

TEST_CASE("item() requires a 0-dimensional tensor", "[tensor]") {
    Tensor t({2, 2});

    CHECK_THROWS_AS(t.item(), std::out_of_range);
    CHECK_THROWS_AS(t[0].item(), std::out_of_range);
}

TEST_CASE("assigning a double writes through to the storage", "[tensor]") {
    Tensor t({2, 2});

    t[0][1] = 9.0;

    CHECK(t[0][1].item() == 9.0);
    CHECK(t[0][0].item() == 0.0);  // neighbouring elements untouched
}

TEST_CASE("a double cannot be assigned to a non-scalar tensor", "[tensor]") {
    Tensor t({2, 2});

    CHECK_THROWS_AS(t[0] = 1.0, std::out_of_range);
}

TEST_CASE("assigning a tensor copies values rather than rebinding", "[tensor]") {
    Tensor destination({2}, {1.0, 2.0});
    Tensor source({2}, {8.0, 9.0});

    destination = source;
    source[0] = 100.0;  // the two must not share storage afterwards

    CHECK(destination[0].item() == 8.0);
    CHECK(destination[1].item() == 9.0);
}

TEST_CASE("assigning a tensor requires a matching shape", "[tensor]") {
    Tensor t({2, 2});

    CHECK_THROWS_AS(t[0] = t, std::invalid_argument);
    CHECK_THROWS_AS(t = Tensor({3, 3}), std::invalid_argument);
}

TEST_CASE("copying between views of one storage handles overlap", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});

    t[1] = t[0];

    CHECK(t[1][0].item() == 1.0);
    CHECK(t[1][1].item() == 2.0);
    CHECK(t[0][0].item() == 1.0);  // the source is unchanged
    CHECK(t[0][1].item() == 2.0);
}

TEST_CASE("self-assignment leaves a tensor unchanged", "[tensor]") {
    Tensor t({2}, {1.0, 2.0});

    t = t;

    CHECK(t[0].item() == 1.0);
    CHECK(t[1].item() == 2.0);
}

TEST_CASE("the copy constructor shares storage", "[tensor]") {
    Tensor t({2}, {1.0, 2.0});
    Tensor copy(t);

    copy[0] = 50.0;

    CHECK(t[0].item() == 50.0);
}

TEST_CASE("a view keeps its storage alive after the parent is gone", "[tensor]") {
    Tensor row = Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0})[1];

    CHECK(row[0].item() == 3.0);
    CHECK(row[1].item() == 4.0);
}

TEST_CASE("repr prints the values", "[tensor]") {
    CHECK(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}).repr() ==
          "Tensor([[1, 2],\n        [3, 4]])");
    CHECK(Tensor({3}, {1.0, 2.0, 3.0}).repr() == "Tensor([1, 2, 3])");
    CHECK(Tensor({}, {5.0}).repr() == "Tensor(5)");
    CHECK(Tensor({0}).repr() == "Tensor([])");
}

TEST_CASE("repr summarises a large tensor and names its shape", "[tensor]") {
    const std::string repr = Tensor({2000}, 1.0).repr();

    CHECK(repr.find("...") != std::string::npos);
    CHECK(repr.find("shape=[2000]") != std::string::npos);
}
