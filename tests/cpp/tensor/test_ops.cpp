#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "slodl/tensor/ops.hpp"
#include "slodl/tensor/tensor.hpp"

TEST_CASE("element_count multiplies the dimensions", "[tensor]") {
    CHECK(element_count({2, 3, 4}) == 24);
    CHECK(element_count({5}) == 5);
    CHECK(element_count({}) == 1);
    CHECK(element_count({3, 0}) == 0);
}

TEST_CASE("format_shape prints the dimensions", "[tensor]") {
    CHECK(format_shape({2, 3}) == "[2, 3]");
    CHECK(format_shape({}) == "[]");
}

TEST_CASE("add_kernel adds element by element", "[tensor]") {
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({2, 2}, {10.0, 20.0, 30.0, 40.0});

    Tensor sum = add_kernel(a, b);

    CHECK(sum.shape() == std::vector<std::size_t>{2, 2});
    CHECK(sum[0][0].item() == 11.0);
    CHECK(sum[0][1].item() == 22.0);
    CHECK(sum[1][0].item() == 33.0);
    CHECK(sum[1][1].item() == 44.0);
}

TEST_CASE("add_kernel returns a fresh contiguous tensor", "[tensor]") {
    Tensor a({2, 2}, 1.0);
    Tensor b({2, 2}, 2.0);

    Tensor sum = add_kernel(a, b);

    CHECK(sum.element_strides() == std::vector<std::size_t>{2, 1});
    CHECK(sum.data() != a.data());
    CHECK(sum.data() != b.data());

    a[0][0].fill_(100.0);
    CHECK(sum[0][0].item() == 3.0);
}

TEST_CASE("add_kernel walks views by their strides", "[tensor]") {
    Tensor matrix({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    Tensor sum = add_kernel(matrix[1], matrix[2]);

    CHECK(sum.shape() == std::vector<std::size_t>{2});
    CHECK(sum[0].item() == 8.0);
    CHECK(sum[1].item() == 10.0);
}

TEST_CASE("add_kernel adds a view to a contiguous tensor", "[tensor]") {
    Tensor matrix({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor row({3}, {10.0, 20.0, 30.0});

    Tensor sum = add_kernel(matrix[1], row);

    CHECK(sum[0].item() == 14.0);
    CHECK(sum[1].item() == 25.0);
    CHECK(sum[2].item() == 36.0);
}

TEST_CASE("add_kernel handles 0-dimensional tensors", "[tensor]") {
    Tensor sum = add_kernel(Tensor({}, 2.0), Tensor({}, 5.0));

    CHECK(sum.shape().empty());
    CHECK(sum.item() == 7.0);
}

TEST_CASE("add_kernel handles an empty tensor", "[tensor]") {
    Tensor sum = add_kernel(Tensor({0}), Tensor({0}));

    CHECK(sum.shape() == std::vector<std::size_t>{0});
}

TEST_CASE("add_kernel walks every element of a 3-dimensional tensor",
          "[tensor]") {
    Tensor a({2, 3, 2}, 1.0);
    Tensor b({2, 3, 2}, 2.0);

    Tensor sum = add_kernel(a, b);

    for (std::size_t i = 0; i < 2; i++) {
        for (std::size_t j = 0; j < 3; j++) {
            for (std::size_t k = 0; k < 2; k++) {
                CHECK(sum[i][j][k].item() == 3.0);
            }
        }
    }
}

TEST_CASE("add_kernel rejects mismatched shapes", "[tensor]") {
    CHECK_THROWS_AS(add_kernel(Tensor({2, 2}), Tensor({2, 3})),
                    std::invalid_argument);
    CHECK_THROWS_AS(add_kernel(Tensor({2}), Tensor({})),
                    std::invalid_argument);
}

TEST_CASE("add_kernel records no autograd history", "[tensor]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    Tensor sum = add_kernel(a, b);

    CHECK_FALSE(sum.requires_grad());
    CHECK(sum.is_leaf());
}

TEST_CASE("elementwise runs any operation", "[tensor]") {
    Tensor a({3}, {1.0, 2.0, 3.0});
    Tensor b({3}, {10.0, 20.0, 30.0});

    Tensor product = elementwise(a, b, [](double left, double right) {
        return left * right;
    });
    Tensor difference = elementwise(b, a, [](double left, double right) {
        return left - right;
    });

    CHECK(product[2].item() == 90.0);
    CHECK(difference[2].item() == 27.0);
}

TEST_CASE("reduce_all folds every element", "[tensor]") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});

    CHECK(reduce_all(t, 0.0, [](double acc, double x) { return acc + x; }) == 10.0);
    CHECK(reduce_all(t, 1.0, [](double acc, double x) { return acc * x; }) == 24.0);
}

TEST_CASE("reduce_all walks a view by its strides", "[tensor]") {
    Tensor t({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    CHECK(reduce_all(t[1], 0.0, [](double acc, double x) { return acc + x; }) == 7.0);
}

TEST_CASE("reduce_all returns the initial value for an empty tensor",
          "[tensor]") {
    CHECK(reduce_all(Tensor({0}), 5.0, [](double acc, double x) {
        return acc + x;
    }) == 5.0);
}

TEST_CASE("sum_kernel adds up every element", "[tensor]") {
    Tensor total = sum_kernel(Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    CHECK(total.shape().empty());
    CHECK(total.item() == 21.0);
}

TEST_CASE("sum_kernel handles views, scalars and empty tensors", "[tensor]") {
    Tensor matrix({2, 2}, {1.0, 2.0, 3.0, 4.0});

    CHECK(sum_kernel(matrix[1]).item() == 7.0);
    CHECK(sum_kernel(Tensor({}, 5.0)).item() == 5.0);
    CHECK(sum_kernel(Tensor({0})).item() == 0.0);
}

TEST_CASE("sum_kernel records no autograd history", "[tensor]") {
    Tensor t({2}, 1.0);
    t.requires_grad_();

    CHECK_FALSE(sum_kernel(t).requires_grad());
}
