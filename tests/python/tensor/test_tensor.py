import numpy as np
import pytest

from slodl import Tensor, full, zeros


# --- construction -----------------------------------------------------------

def test_nested_lists_give_the_nested_shape():
    t = Tensor([[1, 2], [3, 4]])

    assert t.shape == [2, 2]
    assert [t[0][0], t[0][1], t[1][0], t[1][1]] == [1.0, 2.0, 3.0, 4.0]


def test_a_flat_list_gives_a_one_dimensional_tensor():
    # The argument is the data, not the dimensions: this is [2.0, 3.0],
    # not a 2x3 tensor. zeros([2, 3]) is how you ask for the latter.
    t = Tensor([2, 3])

    assert t.shape == [2]
    assert [t[0], t[1]] == [2.0, 3.0]


def test_three_dimensional_nesting():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    assert t.shape == [2, 2, 2]
    assert t[1][0][1] == 6.0


def test_a_bare_number_gives_a_zero_dimensional_tensor():
    t = Tensor(5.0)

    assert t.shape == []
    assert t.item() == 5.0


def test_integers_are_converted_to_float():
    assert isinstance(Tensor([1, 2])[0], float)


def test_ragged_data_is_rejected():
    with pytest.raises(ValueError):
        Tensor([[1, 2], [3]])


def test_data_that_is_not_numeric_is_rejected():
    with pytest.raises(ValueError):
        Tensor([["a", "b"]])


def test_constructing_from_a_tensor_copies():
    original = Tensor([[1, 2], [3, 4]])
    copy = Tensor(original)

    copy[0][0] = 99.0

    assert original[0][0] == 1.0


def test_clone_has_its_own_storage():
    t = Tensor([[1, 2], [3, 4]])
    copy = t.clone()

    copy[0][0] = 99.0

    assert t[0][0] == 1.0
    assert copy[0][0] == 99.0


def test_cloning_a_view_gives_the_view_shape():
    t = Tensor([[1, 2], [3, 4]])

    row = t[1].clone()

    assert row.shape == [2]
    assert [row[0], row[1]] == [3.0, 4.0]


# --- shape, len and repr ----------------------------------------------------

def test_len_is_the_outermost_dimension():
    assert len(zeros([4, 2])) == 4


def test_len_of_a_scalar_is_a_type_error():
    with pytest.raises(TypeError):
        len(Tensor(0.0))


def test_repr_shows_the_values():
    assert repr(Tensor([[1, 2], [3, 4]])) == "Tensor([[1, 2],\n        [3, 4]])"
    assert repr(Tensor([1, 2, 3])) == "Tensor([1, 2, 3])"
    assert repr(Tensor(5.0)) == "Tensor(5)"


def test_repr_summarises_a_large_tensor():
    text = repr(full([2000], 1.0))

    assert "..." in text
    assert "shape=[2000]" in text


# --- indexing ---------------------------------------------------------------

def test_indexing_the_last_dimension_gives_a_float():
    t = Tensor([[1, 2], [3, 4]])

    assert isinstance(t[1][0], float)
    assert t[1][0] == 3.0


def test_indexing_a_middle_dimension_gives_a_tensor():
    t = Tensor([[1, 2], [3, 4]])

    assert isinstance(t[0], Tensor)
    assert t[0].shape == [2]


def test_negative_indices_count_from_the_end():
    t = Tensor([[1, 2], [3, 4]])

    assert t[-1][-1] == 4.0
    assert t[-2][0] == 1.0


def test_out_of_range_index_raises():
    t = zeros([2, 2])

    with pytest.raises(IndexError):
        t[2]
    with pytest.raises(IndexError):
        t[-3]


def test_a_scalar_cannot_be_indexed():
    with pytest.raises(IndexError):
        Tensor(1.0)[0]


def test_item_requires_a_scalar():
    with pytest.raises(IndexError):
        zeros([2, 2]).item()


def test_iteration_terminates():
    # Relies on __getitem__ raising IndexError, not some other error.
    t = Tensor([1, 2, 3])

    assert list(t) == [1.0, 2.0, 3.0]


# --- views ------------------------------------------------------------------

def test_a_view_shares_storage_with_its_parent():
    t = Tensor([[1, 2], [3, 4]])
    row = t[1]

    row[0] = 30.0

    assert t[1][0] == 30.0


def test_indexing_leaves_the_parent_unchanged():
    t = Tensor([[1, 2], [3, 4]])

    t[0]

    assert t.shape == [2, 2]


def test_a_view_outlives_the_tensor_it_came_from():
    row = Tensor([[1, 2], [3, 4]])[1]

    assert [row[0], row[1]] == [3.0, 4.0]


# --- assignment -------------------------------------------------------------

def test_assigning_a_float_writes_through():
    t = zeros([2, 2])

    t[0][1] = 9.0

    assert t[0][1] == 9.0
    assert t[0][0] == 0.0


def test_assigning_a_float_to_a_slice_fills_it():
    t = zeros([2, 2])

    t[0] = 1.0

    assert [t[0][j] for j in range(2)] == [1.0, 1.0]
    assert [t[1][j] for j in range(2)] == [0.0, 0.0]


def test_assigning_a_tensor_copies_values():
    t = Tensor([[1, 2], [3, 4]])
    source = Tensor([8, 9])

    t[0] = source
    source[0] = 100.0  # must not be visible through t

    assert [t[0][0], t[0][1]] == [8.0, 9.0]


def test_assigning_a_tensor_of_the_wrong_shape_raises():
    t = zeros([2, 2])

    with pytest.raises(ValueError):
        t[0] = t


def test_overlapping_copy_between_views_is_correct():
    t = Tensor([[1, 2], [3, 4]])

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


def test_the_constructor_takes_data_only():
    # The old (dims, data) form is gone; shapes come from zeros/ones/full.
    with pytest.raises(TypeError):
        Tensor([2, 2], [1, 2, 3, 4])


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
    t = Tensor([[1, 2], [3, 4]])

    array = np.asarray(t)
    array[0, 0] = 99.0

    assert t[0][0] == 99.0
    assert array.base is not None


def test_asarray_of_a_view():
    t = Tensor([[1, 2], [3, 4]])

    assert np.asarray(t[1]).tolist() == [3.0, 4.0]


def test_asarray_with_another_dtype_copies():
    t = Tensor([1, 2])

    array = np.asarray(t, dtype=np.float32)
    array[0] = 99.0

    assert array.dtype == np.float32
    assert t[0] == 1.0


def test_asarray_of_a_scalar():
    array = np.asarray(Tensor(5.0))

    assert array.shape == ()
    assert array.item() == 5.0


def test_round_trip_through_numpy():
    original = np.arange(6.0).reshape(2, 3)

    assert np.array_equal(np.asarray(Tensor(original)), original)
