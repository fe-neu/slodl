import slodl
from slodl import Tensor, no_grad


def test_recording_is_on_by_default():
    assert slodl.is_grad_enabled() is True


def test_no_grad_stops_recording_inside_the_block():
    a = Tensor([1.0, 2.0]).requires_grad_()

    with no_grad():
        result = a + a
        assert slodl.is_grad_enabled() is False

    assert result.requires_grad is False
    assert result.grad_fn is None
    assert [result[i] for i in range(2)] == [2.0, 4.0]
    assert slodl.is_grad_enabled() is True


def test_no_grad_blocks_nest():
    with no_grad():
        with no_grad():
            assert slodl.is_grad_enabled() is False
        assert slodl.is_grad_enabled() is False

    assert slodl.is_grad_enabled() is True


def test_no_grad_restores_after_an_exception():
    try:
        with no_grad():
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert slodl.is_grad_enabled() is True


def test_no_grad_works_as_a_decorator():
    a = Tensor([1.0, 2.0]).requires_grad_()

    @no_grad()
    def predict(x):
        return x + x

    assert predict(a).requires_grad is False
    assert (a + a).requires_grad is True
