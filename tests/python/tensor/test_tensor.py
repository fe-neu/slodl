import numpy as np
import pytest

from slodl import Tensor


# --- construction -----------------------------------------------------------

def test_dims_only_zero_fills():
    t = Tensor([2, 3])

    assert t.shape == [2, 3]
    assert [t[i][j] for i in range(2) for j in range(3)] == [0.0] * 6


def test_fill_value_fills_every_element():
    t = Tensor([2, 2], 7.0)

    assert [t[0][0], t[0][1], t[1][0], t[1][1]] == [7.0] * 4


def test_data_is_stored_row_major():
    t = Tensor([2, 2], [1, 2, 3, 4])

    assert [t[0][0], t[0][1], t[1][0], t[1][1]] == [1.0, 2.0, 3.0, 4.0]


def test_data_of_the_wrong_length_is_rejected():
    with pytest.raises(IndexError):
        Tensor([2, 2], [1, 2, 3])


def test_zero_dimensional_tensor_holds_one_value():
    t = Tensor([], 5.0)

    assert t.shape == []
    assert t.item() == 5.0


def test_a_list_is_always_read_as_dimensions():
    # Only an ndarray takes the "these are the contents" path.
    assert Tensor([2, 2]).shape == [2, 2]


# --- shape, len and repr ----------------------------------------------------

def test_len_is_the_outermost_dimension():
    assert len(Tensor([4, 2])) == 4


def test_len_of_a_scalar_is_a_type_error():
    with pytest.raises(TypeError):
        len(Tensor([]))


def test_repr_shows_the_values():
    assert repr(Tensor([2, 2], [1, 2, 3, 4])) == "Tensor([[1, 2],\n        [3, 4]])"
    assert repr(Tensor([3], [1, 2, 3])) == "Tensor([1, 2, 3])"
    assert repr(Tensor([], 5.0)) == "Tensor(5)"


def test_repr_summarises_a_large_tensor():
    text = repr(Tensor([2000], 1.0))

    assert "..." in text
    assert "shape=[2000]" in text


# --- indexing ---------------------------------------------------------------

def test_indexing_the_last_dimension_gives_a_float():
    t = Tensor([2, 2], [1, 2, 3, 4])

    assert isinstance(t[1][0], float)
    assert t[1][0] == 3.0


def test_indexing_a_middle_dimension_gives_a_tensor():
    t = Tensor([2, 2], [1, 2, 3, 4])

    assert isinstance(t[0], Tensor)
    assert t[0].shape == [2]


def test_negative_indices_count_from_the_end():
    t = Tensor([2, 2], [1, 2, 3, 4])

    assert t[-1][-1] == 4.0
    assert t[-2][0] == 1.0


def test_out_of_range_index_raises():
    t = Tensor([2, 2])

    with pytest.raises(IndexError):
        t[2]
    with pytest.raises(IndexError):
        t[-3]


def test_a_scalar_cannot_be_indexed():
    with pytest.raises(IndexError):
        Tensor([], 1.0)[0]


def test_item_requires_a_scalar():
    with pytest.raises(IndexError):
        Tensor([2, 2]).item()


def test_iteration_terminates():
    # Relies on __getitem__ raising IndexError, not some other error.
    t = Tensor([3], [1, 2, 3])

    assert list(t) == [1.0, 2.0, 3.0]


# --- views ------------------------------------------------------------------

def test_a_view_shares_storage_with_its_parent():
    t = Tensor([2, 2], [1, 2, 3, 4])
    row = t[1]

    row[0] = 30.0

    assert t[1][0] == 30.0


def test_indexing_leaves_the_parent_unchanged():
    t = Tensor([2, 2], [1, 2, 3, 4])

    t[0]

    assert t.shape == [2, 2]


def test_a_view_outlives_the_tensor_it_came_from():
    row = Tensor([2, 2], [1, 2, 3, 4])[1]

    assert [row[0], row[1]] == [3.0, 4.0]


# --- assignment -------------------------------------------------------------

def test_assigning_a_float_writes_through():
    t = Tensor([2, 2])

    t[0][1] = 9.0

    assert t[0][1] == 9.0
    assert t[0][0] == 0.0


def test_a_float_cannot_be_assigned_to_a_slice():
    t = Tensor([2, 2])

    with pytest.raises(IndexError):
        t[0] = 1.0


def test_assigning_a_tensor_copies_values():
    t = Tensor([2, 2], [1, 2, 3, 4])
    source = Tensor([2], [8, 9])

    t[0] = source
    source[0] = 100.0  # must not be visible through t

    assert [t[0][0], t[0][1]] == [8.0, 9.0]


def test_assigning_a_tensor_of_the_wrong_shape_raises():
    t = Tensor([2, 2])

    with pytest.raises(ValueError):
        t[0] = t


def test_overlapping_copy_between_views_is_correct():
    t = Tensor([2, 2], [1, 2, 3, 4])

    t[1] = t[0]

    assert [t[1][0], t[1][1]] == [1.0, 2.0]
    assert [t[0][0], t[0][1]] == [1.0, 2.0]


# --- numpy ------------------------------------------------------------------

def test_constructing_from_an_array():
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert t.shape == [2, 2]
    assert t[1][0] == 3.0


def test_constructing_from_an_array_copies():
    array = np.array([1.0, 2.0])
    t = Tensor(array)

    array[0] = 99.0

    assert t[0] == 1.0


def test_an_array_and_data_together_is_an_error():
    with pytest.raises(TypeError):
        Tensor(np.array([1.0]), 2.0)


def test_from_numpy_accepts_anything_array_like():
    t = Tensor.from_numpy([[1, 2], [3, 4]])

    assert t.shape == [2, 2]
    assert t[1][1] == 4.0


@pytest.mark.parametrize(
    "array",
    [
        np.array([[1, 2], [3, 4]], dtype=np.int32),          # wrong dtype
        np.array([[1.0, 2.0], [3.0, 4.0]], order="F"),        # column-major
        np.arange(8.0).reshape(4, 2)[::2],                    # strided slice
    ],
)
def test_awkward_layouts_are_converted(array):
    t = Tensor(array)

    assert t.shape == list(array.shape)
    assert np.asarray(t).tolist() == array.tolist()


def test_asarray_shares_memory():
    t = Tensor([2, 2], [1, 2, 3, 4])

    array = np.asarray(t)
    array[0, 0] = 99.0

    assert t[0][0] == 99.0
    assert array.base is not None


def test_asarray_of_a_view():
    t = Tensor([2, 2], [1, 2, 3, 4])

    assert np.asarray(t[1]).tolist() == [3.0, 4.0]


def test_asarray_with_another_dtype_copies():
    t = Tensor([2], [1, 2])

    array = np.asarray(t, dtype=np.float32)
    array[0] = 99.0

    assert array.dtype == np.float32
    assert t[0] == 1.0


def test_asarray_of_a_scalar():
    array = np.asarray(Tensor([], 5.0))

    assert array.shape == ()
    assert array.item() == 5.0


def test_round_trip_through_numpy():
    original = np.arange(6.0).reshape(2, 3)

    assert np.array_equal(np.asarray(Tensor(original)), original)
