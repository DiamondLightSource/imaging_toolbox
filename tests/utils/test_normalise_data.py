import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imaging_toolbox.utils import normalise_data


def test_2d_amplitude():
    arr = 25 * np.ones((5, 5))
    out = normalise_data(arr, dtype="amplitude")
    assert_array_equal(out, np.ones((5, 5)))


def test_2d_phase():
    arr = 25 * np.ones((5, 5))
    out = normalise_data(arr, dtype="phase")
    assert_array_equal(out, np.zeros((5, 5)))


def test_3d_amplitude():
    arr = 25 * np.ones((5, 5, 5))
    out = normalise_data(arr, dtype="amplitude")
    assert_array_equal(out, np.ones((5, 5, 5)))


def test_3d_phase():
    arr = 25 * np.ones((5, 5, 5))
    out = normalise_data(arr, dtype="phase")
    assert_array_equal(out, np.zeros((5, 5, 5)))


def test_incorrect_dtype():
    arr = np.arange(25).reshape(5, 5)
    with pytest.raises(ValueError):
        normalise_data(arr, (0, -1), (0, -1), "blah")
