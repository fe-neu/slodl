#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "slodl/autograd/functions.hpp"
#include "slodl/autograd/grad_mode.hpp"
#include "slodl/autograd/ops.hpp"
#include "slodl/tensor/ops.hpp"
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

TEST_CASE("mul multiplies element by element", "[autograd]") {
    Tensor product = mul(Tensor({2}, {2.0, 3.0}), Tensor({2}, {10.0, 20.0}));

    CHECK(product[0].item() == 20.0);
    CHECK(product[1].item() == 60.0);
}

TEST_CASE("mul records a graph when an input requires a gradient",
          "[autograd]") {
    Tensor a({2}, 2.0);
    a.requires_grad_();

    Tensor product = mul(a, Tensor({2}, 3.0));

    REQUIRE(product.autograd_meta()->grad_fn != nullptr);
    CHECK(product.autograd_meta()->grad_fn->name == "MulBackward");
}

TEST_CASE("operator* multiplies like mul", "[autograd]") {
    Tensor a({2}, 2.0);
    a.requires_grad_();

    Tensor product = a * Tensor({2}, 4.0);

    CHECK(product[0].item() == 8.0);
    CHECK(product.autograd_meta()->grad_fn->name == "MulBackward");
}

TEST_CASE("MulBackward scales the gradient by the opposite input",
          "[autograd]") {
    Tensor a({2}, {2.0, 3.0});
    Tensor b({2}, {10.0, 20.0});
    a.requires_grad_();
    b.requires_grad_();

    const std::shared_ptr<Node> node = mul(a, b).autograd_meta()->grad_fn;
    std::vector<std::optional<Tensor>> gradients =
        node->apply({Tensor({2}, {1.0, 2.0})});

    REQUIRE(gradients.size() == 2);
    CHECK((*gradients[0])[0].item() == 10.0);   // 1 * b[0]
    CHECK((*gradients[0])[1].item() == 40.0);   // 2 * b[1]
    CHECK((*gradients[1])[0].item() == 2.0);    // 1 * a[0]
    CHECK((*gradients[1])[1].item() == 6.0);    // 2 * a[1]
}

TEST_CASE("backward through mul gives each input the other's value",
          "[autograd]") {
    Tensor a({}, 3.0);
    Tensor b({}, 4.0);
    a.requires_grad_();
    b.requires_grad_();

    mul(a, b).backward();

    CHECK(a.grad()->item() == 4.0);
    CHECK(b.grad()->item() == 3.0);
}

TEST_CASE("mul survives the inputs going out of scope", "[autograd]") {
    Tensor a({}, 3.0);
    a.requires_grad_();

    Tensor product = mul(a, Tensor({}, 4.0));
    product.backward();

    CHECK(a.grad()->item() == 4.0);
}

TEST_CASE("a saved input carries no history", "[autograd]") {
    Tensor a({}, 3.0);
    Tensor b({}, 4.0);
    a.requires_grad_();
    b.requires_grad_();

    Tensor first = mul(a, b);
    // Multiplying the result again must not reach a or b through what
    // MulBackward saved, only through its edges.
    mul(first, Tensor({}, 2.0)).backward();

    CHECK(a.grad()->item() == 8.0);
    CHECK(b.grad()->item() == 6.0);
}

TEST_CASE("mixing add and mul follows the product rule", "[autograd]") {
    Tensor a({}, 2.0);
    Tensor b({}, 5.0);
    a.requires_grad_();
    b.requires_grad_();

    // (a + b) * a, so d/da = (a + b) + a = 9, d/db = a = 2
    mul(add(a, b), a).backward();

    CHECK(a.grad()->item() == 9.0);
    CHECK(b.grad()->item() == 2.0);
}

TEST_CASE("sum adds up every element", "[autograd]") {
    Tensor total = sum(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}));

    CHECK(total.shape().empty());
    CHECK(total.item() == 10.0);
}

TEST_CASE("sum records a graph when its input requires a gradient",
          "[autograd]") {
    Tensor a({2}, 1.0);
    a.requires_grad_();

    Tensor total = sum(a);

    REQUIRE(total.autograd_meta()->grad_fn != nullptr);
    CHECK(total.autograd_meta()->grad_fn->name == "SumBackward");
    CHECK(total.autograd_meta()->grad_fn->next_edges.size() == 1);
}

TEST_CASE("backward through sum gives every element a gradient of one",
          "[autograd]") {
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    a.requires_grad_();

    sum(a).backward();

    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->shape() == std::vector<std::size_t>{2, 2});
    CHECK((*a.grad())[0][0].item() == 1.0);
    CHECK((*a.grad())[1][1].item() == 1.0);
}

TEST_CASE("sum expands whatever gradient it receives", "[autograd]") {
    Tensor a({3}, 1.0);
    a.requires_grad_();

    const std::shared_ptr<Node> node = sum(a).autograd_meta()->grad_fn;
    std::vector<std::optional<Tensor>> gradients = node->apply({Tensor({}, 2.0)});

    REQUIRE(gradients.size() == 1);
    CHECK((*gradients[0]).shape() == std::vector<std::size_t>{3});
    CHECK((*gradients[0])[0].item() == 2.0);
    CHECK((*gradients[0])[2].item() == 2.0);
}

TEST_CASE("sum makes a non-scalar graph differentiable", "[autograd]") {
    Tensor a({2}, {2.0, 3.0});
    Tensor b({2}, {10.0, 20.0});
    a.requires_grad_();
    b.requires_grad_();

    // sum(a * b), so d/da = b and d/db = a.
    sum(mul(a, b)).backward();

    CHECK((*a.grad())[0].item() == 10.0);
    CHECK((*a.grad())[1].item() == 20.0);
    CHECK((*b.grad())[0].item() == 2.0);
    CHECK((*b.grad())[1].item() == 3.0);
}

TEST_CASE("sum of a view gradient reaches only the view's elements",
          "[autograd]") {
    Tensor a({2}, {1.0, 2.0});
    a.requires_grad_();

    sum(add(a, a)).backward();

    CHECK((*a.grad())[0].item() == 2.0);
    CHECK((*a.grad())[1].item() == 2.0);
}

TEST_CASE("sub subtracts element by element", "[autograd]") {
    Tensor difference = sub(Tensor({2}, {10.0, 3.0}), Tensor({2}, {4.0, 8.0}));

    CHECK(difference[0].item() == 6.0);
    CHECK(difference[1].item() == -5.0);
}

TEST_CASE("sub records a graph when an input requires a gradient",
          "[autograd]") {
    Tensor a({2}, 1.0);
    a.requires_grad_();

    Tensor difference = sub(a, Tensor({2}, 2.0));

    REQUIRE(difference.autograd_meta()->grad_fn != nullptr);
    CHECK(difference.autograd_meta()->grad_fn->name == "SubBackward");
}

TEST_CASE("SubBackward negates only the right operand's gradient",
          "[autograd]") {
    Tensor a({2}, 1.0);
    Tensor b({2}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    const std::shared_ptr<Node> node = sub(a, b).autograd_meta()->grad_fn;
    std::vector<std::optional<Tensor>> gradients =
        node->apply({Tensor({2}, {3.0, 4.0})});

    REQUIRE(gradients.size() == 2);
    CHECK((*gradients[0])[0].item() == 3.0);
    CHECK((*gradients[0])[1].item() == 4.0);
    CHECK((*gradients[1])[0].item() == -3.0);
    CHECK((*gradients[1])[1].item() == -4.0);
}

TEST_CASE("backward through sub gives 1 and -1", "[autograd]") {
    Tensor a({}, 5.0);
    Tensor b({}, 3.0);
    a.requires_grad_();
    b.requires_grad_();

    sub(a, b).backward();

    CHECK(a.grad()->item() == 1.0);
    CHECK(b.grad()->item() == -1.0);
}

TEST_CASE("operator- subtracts like sub", "[autograd]") {
    Tensor a({2}, 5.0);
    a.requires_grad_();

    Tensor difference = a - Tensor({2}, 2.0);

    CHECK(difference[0].item() == 3.0);
    CHECK(difference.autograd_meta()->grad_fn->name == "SubBackward");
}

TEST_CASE("neg flips every sign", "[autograd]") {
    Tensor negated = neg(Tensor({2}, {1.0, -2.0}));

    CHECK(negated[0].item() == -1.0);
    CHECK(negated[1].item() == 2.0);
}

TEST_CASE("neg records a graph and negates the gradient", "[autograd]") {
    Tensor a({}, 3.0);
    a.requires_grad_();

    Tensor negated = neg(a);
    REQUIRE(negated.autograd_meta()->grad_fn != nullptr);
    CHECK(negated.autograd_meta()->grad_fn->name == "NegBackward");

    negated.backward();
    CHECK(a.grad()->item() == -1.0);
}

TEST_CASE("unary operator- negates like neg", "[autograd]") {
    Tensor a({2}, 2.0);
    a.requires_grad_();

    Tensor negated = -a;

    CHECK(negated[0].item() == -2.0);
    CHECK(negated.autograd_meta()->grad_fn->name == "NegBackward");
}

TEST_CASE("subtracting a tensor from itself gives a zero gradient",
          "[autograd]") {
    Tensor a({}, 4.0);
    a.requires_grad_();

    sub(a, a).backward();

    // +1 from the left operand, -1 from the right.
    CHECK(a.grad()->item() == 0.0);
}

TEST_CASE("sub and neg agree with each other", "[autograd]") {
    Tensor a({2}, {5.0, 6.0});
    Tensor b({2}, {2.0, 3.0});
    a.requires_grad_();
    b.requires_grad_();

    // sum(a - b) and sum(a + (-b)) must give the same gradients.
    sum(sub(a, b)).backward();
    const double a_direct = (*a.grad())[0].item();
    const double b_direct = (*b.grad())[0].item();

    Tensor c({2}, {5.0, 6.0});
    Tensor d({2}, {2.0, 3.0});
    c.requires_grad_();
    d.requires_grad_();
    sum(add(c, neg(d))).backward();

    CHECK((*c.grad())[0].item() == a_direct);
    CHECK((*d.grad())[0].item() == b_direct);
}

TEST_CASE("div divides element by element", "[autograd]") {
    Tensor quotient = div(Tensor({2}, {6.0, 9.0}), Tensor({2}, {2.0, 3.0}));

    CHECK(quotient[0].item() == 3.0);
    CHECK(quotient[1].item() == 3.0);
}

TEST_CASE("div records a graph when an input requires a gradient",
          "[autograd]") {
    Tensor a({2}, 6.0);
    a.requires_grad_();

    Tensor quotient = div(a, Tensor({2}, 2.0));

    REQUIRE(quotient.autograd_meta()->grad_fn != nullptr);
    CHECK(quotient.autograd_meta()->grad_fn->name == "DivBackward");
}

TEST_CASE("backward through div follows the quotient rule", "[autograd]") {
    Tensor a({}, 6.0);
    Tensor b({}, 2.0);
    a.requires_grad_();
    b.requires_grad_();

    div(a, b).backward();

    // d/da (a/b) = 1/b = 0.5; d/db (a/b) = -a/b^2 = -1.5
    CHECK(a.grad()->item() == 0.5);
    CHECK(b.grad()->item() == -1.5);
}

TEST_CASE("the divisor's gradient is negative", "[autograd]") {
    Tensor a({2}, {8.0, 3.0});
    Tensor b({2}, {4.0, 1.0});
    a.requires_grad_();
    b.requires_grad_();

    sum(div(a, b)).backward();

    CHECK((*a.grad())[0].item() == 0.25);    // 1/4
    CHECK((*a.grad())[1].item() == 1.0);     // 1/1
    CHECK((*b.grad())[0].item() == -0.5);    // -8/16
    CHECK((*b.grad())[1].item() == -3.0);    // -3/1
}

TEST_CASE("div matches a finite-difference gradient", "[autograd]") {
    const double step = 1e-6;

    Tensor a({}, 7.0);
    Tensor b({}, 3.0);
    a.requires_grad_();
    b.requires_grad_();
    div(a, b).backward();

    const double numeric_a =
        (div_kernel(Tensor({}, 7.0 + step), Tensor({}, 3.0)).item() -
         div_kernel(Tensor({}, 7.0 - step), Tensor({}, 3.0)).item()) / (2 * step);
    const double numeric_b =
        (div_kernel(Tensor({}, 7.0), Tensor({}, 3.0 + step)).item() -
         div_kernel(Tensor({}, 7.0), Tensor({}, 3.0 - step)).item()) / (2 * step);

    CHECK(a.grad()->item() == Catch::Approx(numeric_a).epsilon(1e-6));
    CHECK(b.grad()->item() == Catch::Approx(numeric_b).epsilon(1e-6));
}

TEST_CASE("operator/ divides like div", "[autograd]") {
    Tensor a({2}, 6.0);
    a.requires_grad_();

    Tensor quotient = a / Tensor({2}, 2.0);

    CHECK(quotient[0].item() == 3.0);
    CHECK(quotient.autograd_meta()->grad_fn->name == "DivBackward");
}

TEST_CASE("dividing a tensor by itself gives a zero gradient", "[autograd]") {
    Tensor a({}, 5.0);
    a.requires_grad_();

    div(a, a).backward();

    // 1/a from the dividend and -a/a^2 from the divisor cancel.
    CHECK(a.grad()->item() == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("expand records a graph and sums the gradient back", "[autograd]") {
    Tensor row({3}, {1.0, 2.0, 3.0});
    row.requires_grad_();

    Tensor wide = expand(row, {2, 3});

    REQUIRE(wide.autograd_meta()->grad_fn != nullptr);
    CHECK(wide.autograd_meta()->grad_fn->name == "ExpandBackward");

    sum(wide).backward();

    // Each element was read once per row, so each gets a gradient of 2.
    REQUIRE(row.grad() != nullptr);
    CHECK(row.grad()->shape() == std::vector<std::size_t>{3});
    CHECK((*row.grad())[0].item() == 2.0);
    CHECK((*row.grad())[2].item() == 2.0);
}

TEST_CASE("add broadcasts a row across a matrix", "[autograd]") {
    Tensor matrix({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor row({3}, {10.0, 20.0, 30.0});

    Tensor result = add(matrix, row);

    CHECK(result.shape() == std::vector<std::size_t>{2, 3});
    CHECK(result[0][0].item() == 11.0);
    CHECK(result[1][2].item() == 36.0);
}

TEST_CASE("a broadcast operand's gradient is summed back to its shape",
          "[autograd]") {
    Tensor matrix({2, 3}, 1.0);
    Tensor bias({3}, 1.0);
    matrix.requires_grad_();
    bias.requires_grad_();

    sum(add(matrix, bias)).backward();

    CHECK(matrix.grad()->shape() == std::vector<std::size_t>{2, 3});
    CHECK((*matrix.grad())[0][0].item() == 1.0);

    // The bias took part in both rows, so its gradient is 2 per element.
    CHECK(bias.grad()->shape() == std::vector<std::size_t>{3});
    CHECK((*bias.grad())[0].item() == 2.0);
    CHECK((*bias.grad())[2].item() == 2.0);
}

TEST_CASE("multiplying by a broadcast scalar scales and sums", "[autograd]") {
    Tensor values({3}, {1.0, 2.0, 3.0});
    Tensor factor({}, 10.0);
    values.requires_grad_();
    factor.requires_grad_();

    sum(mul(values, factor)).backward();

    // d/dvalues = factor, d/dfactor = sum(values)
    CHECK((*values.grad())[0].item() == 10.0);
    CHECK(values.grad()->shape() == std::vector<std::size_t>{3});
    CHECK(factor.grad()->shape().empty());
    CHECK(factor.grad()->item() == 6.0);
}

TEST_CASE("broadcasting works in both directions at once", "[autograd]") {
    Tensor column({2, 1}, {1.0, 2.0});
    Tensor row({1, 3}, {10.0, 20.0, 30.0});
    column.requires_grad_();
    row.requires_grad_();

    Tensor product = mul(column, row);
    CHECK(product.shape() == std::vector<std::size_t>{2, 3});
    CHECK(product[1][2].item() == 60.0);

    sum(product).backward();

    // Each column element multiplies the whole row, so its gradient is the
    // row's total, and vice versa.
    CHECK(column.grad()->shape() == std::vector<std::size_t>{2, 1});
    CHECK((*column.grad())[0][0].item() == 60.0);
    CHECK(row.grad()->shape() == std::vector<std::size_t>{1, 3});
    CHECK((*row.grad())[0][0].item() == 3.0);
}

TEST_CASE("broadcast division matches a finite-difference gradient",
          "[autograd]") {
    const double step = 1e-6;

    Tensor values({2}, {6.0, 8.0});
    Tensor divisor({}, 2.0);
    values.requires_grad_();
    divisor.requires_grad_();
    sum(div(values, divisor)).backward();

    const auto loss = [](double d) {
        return sum_kernel(div_kernel(Tensor({2}, {6.0, 8.0}),
                                     Tensor({}, d).expand({2}))).item();
    };
    const double numeric = (loss(2.0 + step) - loss(2.0 - step)) / (2 * step);

    CHECK(divisor.grad()->item() == Catch::Approx(numeric).epsilon(1e-6));
    CHECK((*values.grad())[0].item() == Catch::Approx(0.5));
}

TEST_CASE("incompatible shapes are still rejected", "[autograd]") {
    CHECK_THROWS_AS(add(Tensor({2}), Tensor({3})), std::invalid_argument);
    CHECK_THROWS_AS(mul(Tensor({2, 3}), Tensor({2, 4})), std::invalid_argument);
}

TEST_CASE("mean averages every element", "[autograd]") {
    Tensor average = mean(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}));

    CHECK(average.shape().empty());
    CHECK(average.item() == 2.5);
}

TEST_CASE("mean of an empty tensor is not a number", "[autograd]") {
    const double value = mean(Tensor({0})).item();

    CHECK(value != value);
}

TEST_CASE("backward through mean gives every element 1/n", "[autograd]") {
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    a.requires_grad_();

    mean(a).backward();

    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->shape() == std::vector<std::size_t>{2, 2});
    CHECK((*a.grad())[0][0].item() == Catch::Approx(0.25));
    CHECK((*a.grad())[1][1].item() == Catch::Approx(0.25));
}

TEST_CASE("mean records the graph of the ops it is built from", "[autograd]") {
    Tensor a({3}, 1.0);
    a.requires_grad_();

    Tensor average = mean(a);

    // Composed, not a node of its own: the last op is the division.
    REQUIRE(average.autograd_meta()->grad_fn != nullptr);
    CHECK(average.autograd_meta()->grad_fn->name == "DivBackward");
}

TEST_CASE("mean records nothing inside a NoGradGuard", "[autograd]") {
    Tensor a({2}, 4.0);
    a.requires_grad_();

    NoGradGuard guard;
    Tensor average = mean(a);

    CHECK(average.item() == 4.0);
    CHECK(average.is_leaf());
}

TEST_CASE("mean of a difference is a usable loss", "[autograd]") {
    Tensor prediction({2}, {3.0, 5.0});
    Tensor target({2}, {1.0, 1.0});
    prediction.requires_grad_();

    Tensor error = sub(prediction, target);
    mean(mul(error, error)).backward();

    // d/dp mean((p - t)^2) = 2(p - t)/n
    CHECK((*prediction.grad())[0].item() == Catch::Approx(2.0));
    CHECK((*prediction.grad())[1].item() == Catch::Approx(4.0));
}
