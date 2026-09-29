import pytest

from slodl import Tensor, full, ones, zeros


def test_zeros_fills_with_zero():
    t = zeros([2, 3])

    assert t.shape == [2, 3]
    assert [t[i][j] for i in range(2) for j in range(3)] == [0.0] * 6


def test_ones_fills_with_one():
    t = ones([2, 2])

    assert [t[0][0], t[0][1], t[1][0], t[1][1]] == [1.0] * 4


def test_full_fills_with_the_given_value():
    t = full([2, 2], 7.5)

    assert [t[0][0], t[1][1]] == [7.5, 7.5]


@pytest.mark.parametrize("make", [zeros, ones])
def test_an_empty_shape_gives_a_scalar(make):
    t = make([])

    assert t.shape == []


def test_full_with_an_empty_shape_gives_a_scalar():
    assert full([], 3.0).item() == 3.0


def test_a_one_dimensional_shape():
    assert zeros([4]).shape == [4]


def test_shapes_accept_any_sequence_of_ints():
    assert zeros((2, 3)).shape == [2, 3]
    assert zeros(range(2, 4)).shape == [2, 3]


def test_the_result_is_a_wrapped_tensor():
    # Not a raw _core.Tensor: the Python API must come back out.
    assert isinstance(zeros([2]), Tensor)


def test_each_call_gets_its_own_storage():
    a = zeros([2])
    b = zeros([2])

    a[0] = 5.0

    assert b[0] == 0.0
