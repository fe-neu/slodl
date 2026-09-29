import pytest

from slodl import Tensor


def test_a_fresh_tensor_tracks_nothing():
    t = Tensor([1.0, 2.0])

    assert t.requires_grad is False
    assert t.is_leaf is True
    assert t.grad is None
    assert t.grad_fn is None


def test_requires_grad_can_be_set_as_a_property():
    t = Tensor([1.0, 2.0])
    t.requires_grad = True

    assert t.requires_grad is True

    t.requires_grad = False
    assert t.requires_grad is False


def test_requires_grad_returns_the_tensor_for_chaining():
    t = Tensor([1.0, 2.0])

    assert t.requires_grad_() is t
    assert t.requires_grad is True


def test_requires_grad_is_rejected_on_a_result():
    a = Tensor([1.0, 2.0]).requires_grad_()
    result = a + a

    with pytest.raises(ValueError):
        result.requires_grad = True


def test_detach_drops_the_history_but_shares_the_data():
    a = Tensor([1.0, 2.0]).requires_grad_()

    detached = a.detach()

    assert detached.requires_grad is False
    assert detached.is_leaf is True

    detached[0] = 9.0
    assert a[0] == 9.0


def test_backward_fills_the_gradients_of_the_inputs():
    a = Tensor(2.0).requires_grad_()
    b = Tensor(3.0).requires_grad_()

    (a + b).backward()

    assert a.grad is not None
    assert a.grad.item() == 1.0
    assert b.grad.item() == 1.0


def test_backward_sums_the_gradient_of_a_tensor_used_twice():
    a = Tensor(2.0).requires_grad_()

    (a + a).backward()

    assert a.grad.item() == 2.0


def test_backward_follows_a_chain():
    a = Tensor(1.0).requires_grad_()
    b = Tensor(2.0).requires_grad_()

    (a + b + a).backward()

    assert a.grad.item() == 2.0
    assert b.grad.item() == 1.0


def test_gradients_accumulate_across_passes():
    a = Tensor(1.0).requires_grad_()

    (a + a).backward()
    (a + a).backward()

    assert a.grad.item() == 4.0


def test_backward_needs_a_scalar_that_requires_a_gradient():
    vector_valued = Tensor([1.0, 2.0]).requires_grad_()
    with pytest.raises(ValueError):
        (vector_valued + vector_valued).backward()

    with pytest.raises(ValueError):
        Tensor(1.0).backward()


def test_gradients_carry_no_history_of_their_own():
    a = Tensor(1.0).requires_grad_()

    (a + a).backward()

    assert a.grad.requires_grad is False
    assert a.grad.grad_fn is None
