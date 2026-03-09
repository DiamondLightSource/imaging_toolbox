from typing import cast

import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation  # type: ignore


def fft_shift_data(data: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Function that shifts an image in Fourier space

    Parameters
    ----------
    data: ndarray
        2D image to shift
    shift: ndarray
        Array containing values for shifting y and x axes

    Returns:
    out: ndarray
        The 2D image after it has been shifted
    """
    return np.fft.ifftn(fourier_shift(np.fft.fftn(data), shift)).real


def align_data(
    data: np.ndarray,
    reference: np.ndarray,
    space: str = "real",
    upsample_factor: int = 100,
    normalisation: str = "phase",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to align an image or a stack of image with respect to a reference image.

    Parameters
    ----------
    data: ndarray
        A 2D or 3D array of images to align with reference.
    reference: ndarray
        Reference image
    space: str, one of "real" or "fourier", optional
        Defines how the algorithm interprets the input data.
    upsample_factor: int, optional
        Upsampling factor. Images will be registered to within 1 / upsample_factor.
    normalisation: {"phase", None}
        The type of normalisation to apply to the cross-correlation.

    Returns
    -------
    aligned_data: ndarray
        Array containing aligned image data
    shifts: ndarray
        Array containing image shifts
    """
    if data.ndim == 2:
        shift, _, _ = cast(
            tuple[np.ndarray, float, float],
            phase_cross_correlation(
                reference_image=reference,
                moving_image=data,
                upsample_factor=upsample_factor,
                space=space,
                normalization=normalisation,
            ),
        )
        aligned_data = fft_shift_data(data, shift)
        return aligned_data, shift
    else:
        aligned_data = np.zeros(data.shape)
        shifts = np.zeros((data.shape[0], 2))
        for i in range(data.shape[0]):
            shift, _, _ = cast(
                tuple[np.ndarray, float, float],
                phase_cross_correlation(
                    reference_image=reference,
                    moving_image=data[i],
                    upsample_factor=upsample_factor,
                    space=space,
                    normalization=normalisation,
                ),
            )
            aligned_data[i, :, :] = fft_shift_data(data[i, :, :], shift)
            shifts[i, :] = shift
        return aligned_data, shifts
