import pytest

import slodl
from slodl import Tensor


def test_add_returns_the_sum():
    result = slodl.add(Tensor([1.0, 2.0]), Tensor([10.0, 20.0]))

    assert [result[i] for i in range(2)] == [11.0, 22.0]


def test_operator_adds_like_add():
    result = Tensor([[1.0, 2.0], [3.0, 4.0]]) + Tensor([[10.0, 20.0], [30.0, 40.0]])

    assert result.shape == [2, 2]
    assert result[1][1] == 44.0


def test_add_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]) + Tensor([1.0, 2.0, 3.0])


def test_add_of_untracked_tensors_records_nothing():
    result = Tensor([1.0]) + Tensor([2.0])

    assert result.requires_grad is False
    assert result.is_leaf is True
    assert result.grad_fn is None


def test_add_records_when_an_operand_requires_grad():
    a = Tensor([1.0, 2.0]).requires_grad_()

    result = a + Tensor([3.0, 4.0])

    assert result.requires_grad is True
    assert result.is_leaf is False
    assert result.grad_fn is not None
    assert result.grad_fn.name == "AddBackward"
    assert repr(result.grad_fn) == "<AddBackward>"


def test_adding_a_non_tensor_is_a_type_error():
    with pytest.raises(TypeError):
        Tensor([1.0]) + 1.0


def test_mul_returns_the_elementwise_product():
    result = slodl.mul(Tensor([2.0, 3.0]), Tensor([10.0, 20.0]))

    assert [result[i] for i in range(2)] == [20.0, 60.0]


def test_operator_multiplies_like_mul():
    result = Tensor([[1.0, 2.0], [3.0, 4.0]]) * Tensor([[2.0, 2.0], [3.0, 3.0]])

    assert result.shape == [2, 2]
    assert result[1][1] == 12.0


def test_mul_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]) * Tensor([1.0, 2.0, 3.0])


def test_mul_records_when_an_operand_requires_grad():
    a = Tensor([2.0, 3.0]).requires_grad_()

    result = a * Tensor([4.0, 5.0])

    assert result.requires_grad is True
    assert result.grad_fn.name == "MulBackward"


def test_mul_of_untracked_tensors_records_nothing():
    result = Tensor([2.0]) * Tensor([3.0])

    assert result.requires_grad is False
    assert result.grad_fn is None


def test_backward_through_mul_uses_the_other_operand():
    a = Tensor(3.0).requires_grad_()
    b = Tensor(4.0).requires_grad_()

    (a * b).backward()

    assert a.grad.item() == 4.0
    assert b.grad.item() == 3.0


def test_multiplying_by_a_float_is_a_type_error():
    with pytest.raises(TypeError):
        Tensor([1.0]) * 2.0


def test_sum_adds_up_every_element():
    total = Tensor([[1.0, 2.0], [3.0, 4.0]]).sum()

    assert total.shape == []
    assert total.item() == 10.0


def test_the_free_function_sums_like_the_method():
    t = Tensor([1.0, 2.0, 3.0])

    assert slodl.sum(t).item() == t.sum().item() == 6.0


def test_sum_of_a_view():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert t[1].sum().item() == 7.0


def test_sum_records_when_its_input_requires_grad():
    a = Tensor([1.0, 2.0]).requires_grad_()

    total = a.sum()

    assert total.requires_grad is True
    assert total.grad_fn.name == "SumBackward"


def test_backward_through_sum_gives_every_element_one():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]]).requires_grad_()

    a.sum().backward()

    assert a.grad.shape == [2, 2]
    assert [a.grad[0][j] for j in range(2)] == [1.0, 1.0]
    assert [a.grad[1][j] for j in range(2)] == [1.0, 1.0]


def test_sum_makes_a_vector_computation_differentiable():
    a = Tensor([2.0, 3.0]).requires_grad_()
    b = Tensor([10.0, 20.0]).requires_grad_()

    (a * b).sum().backward()

    assert [a.grad[i] for i in range(2)] == [10.0, 20.0]
    assert [b.grad[i] for i in range(2)] == [2.0, 3.0]


def test_sub_returns_the_differences():
    result = slodl.sub(Tensor([10.0, 3.0]), Tensor([4.0, 8.0]))

    assert [result[i] for i in range(2)] == [6.0, -5.0]


def test_operator_subtracts_like_sub():
    result = Tensor([[5.0, 6.0], [7.0, 8.0]]) - Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert result.shape == [2, 2]
    assert result[1][1] == 4.0


def test_sub_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]) - Tensor([1.0, 2.0, 3.0])


def test_backward_through_sub_negates_the_right_gradient():
    a = Tensor(5.0).requires_grad_()
    b = Tensor(3.0).requires_grad_()

    (a - b).backward()

    assert a.grad.item() == 1.0
    assert b.grad.item() == -1.0


def test_neg_flips_every_sign():
    result = slodl.neg(Tensor([1.0, -2.0, 0.0]))

    assert [result[i] for i in range(3)] == [-1.0, 2.0, 0.0]


def test_unary_operator_negates_like_neg():
    a = Tensor([2.0, 3.0]).requires_grad_()

    result = -a

    assert [result[i] for i in range(2)] == [-2.0, -3.0]
    assert result.grad_fn.name == "NegBackward"


def test_backward_through_neg_negates_the_gradient():
    a = Tensor(3.0).requires_grad_()

    (-a).backward()

    assert a.grad.item() == -1.0


def test_subtracting_a_float_is_a_type_error():
    with pytest.raises(TypeError):
        Tensor([1.0]) - 2.0
