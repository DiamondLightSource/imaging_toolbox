import numpy as np
import pytest

from imaging_toolbox.ptychography import remove_ramp_and_unwrap_phase


def test_dimensions_mismatch():
    data = np.ones((5, 5))
    mask = np.ones((5, 5, 5), dtype="bool")
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, mask)


def test_2d_data_2d_mask_shape_mismatch():
    data = np.ones((5, 5))
    small_mask = np.ones(
        (
            3,
            3,
        ),
        dtype="bool",
    )
    big_mask = np.ones(
        (
            9,
            9,
        ),
        dtype="bool",
    )
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, small_mask)
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, big_mask)


def test_3d_data_2d_mask_shape_mismatch():
    data = np.ones((5, 5, 5))
    small_mask = np.ones(
        (
            3,
            3,
        ),
        dtype="bool",
    )
    big_mask = np.ones(
        (
            9,
            9,
        ),
        dtype="bool",
    )
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, small_mask)
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, big_mask)


def test_3d_data_3d_mask_shape_mismatch():
    data = np.ones((5, 5, 5))
    small_mask = np.ones(
        (
            3,
            3,
            3,
        ),
        dtype="bool",
    )
    big_mask = np.ones(
        (
            9,
            9,
            9,
        ),
        dtype="bool",
    )
    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, small_mask)

    with pytest.raises(ValueError):
        remove_ramp_and_unwrap_phase(data, big_mask)
