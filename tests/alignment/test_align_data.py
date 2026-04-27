from typing import cast

import numpy as np
from numpy.testing import assert_array_equal
from skimage.data import camera

from imaging_toolbox.alignment import align_data, fft_shift_data


def test_alignment():
    ref = cast(np.ndarray, camera())
    shift = np.array([-2.4, 1.32])
    shifted_image = fft_shift_data(ref, shift)
    _, shifts = align_data(shifted_image, ref)
    assert_array_equal(shifts, -shift)
