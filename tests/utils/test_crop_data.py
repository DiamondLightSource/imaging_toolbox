import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from imaging_toolbox.utils import crop_data


def test_2d_both_ranges():
    arr = np.arange(25).reshape(5, 5)
    out = crop_data(arr, y_range=(1, 4), x_range=(1, 4))
    assert_array_equal(out[0, :], [6, 7, 8])
    assert_array_equal(out[:, 0], [6, 11, 16])
    assert_equal(out.shape, (3, 3))


def test_2d_only_y_range():
    arr = np.arange(25).reshape(5, 5)
    out = crop_data(arr, y_range=(1, 4))
    assert_array_equal(out[0, :], [5, 6, 7, 8, 9])
    assert_array_equal(out[:, 0], [5, 10, 15])
    assert_equal(out.shape, (3, 5))


def test_2d_only_x_range():
    arr = np.arange(25).reshape(5, 5)
    out = crop_data(arr, x_range=(1, 4))
    assert_array_equal(out[0, :], [1, 2, 3])
    assert_array_equal(out[:, 0], [1, 6, 11, 16, 21])
    assert_equal(out.shape, (5, 3))


def test_2d_no_ranges():
    arr = np.arange(25).reshape(5, 5)
    out = crop_data(arr)
    assert_array_equal(out[0, :], [0, 1, 2, 3, 4])
    assert_array_equal(out[:, 0], [0, 5, 10, 15, 20])
    assert_equal(out.shape, (5, 5))


def test_3d_both_ranges():
    arr = np.arange(27).reshape(3, 3, 3)
    out = crop_data(arr, y_range=(1, 3), x_range=(1, 3))
    assert_array_equal(out[:, 0, 0], [4, 13, 22])
    assert_array_equal(out[0, :, 0], [4, 7])
    assert_array_equal(out[0, 0, :], [4, 5])
    assert_equal(out.shape, (3, 2, 2))


def test_3d_only_y_range():
    arr = np.arange(27).reshape(3, 3, 3)
    out = crop_data(arr, y_range=(1, 4))
    assert_array_equal(out[:, 0, 0], [3, 12, 21])
    assert_array_equal(out[0, :, 0], [3, 6])
    assert_array_equal(out[0, 0, :], [3, 4, 5])
    assert_equal(out.shape, (3, 2, 3))


def test_3d_only_x_range():
    arr = np.arange(27).reshape(3, 3, 3)
    out = crop_data(arr, x_range=(1, 4))
    assert_array_equal(out[:, 0, 0], [1, 10, 19])
    assert_array_equal(out[0, :, 0], [1, 4, 7])
    assert_array_equal(out[0, 0, :], [1, 2])
    assert_equal(out.shape, (3, 3, 2))


def test_3d_no_ranges():
    arr = np.arange(27).reshape(3, 3, 3)
    out = crop_data(arr)
    assert_array_equal(out[:, 0, 0], [0, 9, 18])
    assert_array_equal(out[0, :, 0], [0, 3, 6])
    assert_array_equal(out[0, 0, :], [0, 1, 2])
    assert_equal(out.shape, (3, 3, 3))
