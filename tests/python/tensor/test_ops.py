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
