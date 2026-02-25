from typing import cast

import numpy as np
from ptypy.utils.plot_utils import rmphaseramp
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


def crop_data(
    data: np.ndarray,
    y_range: tuple[int, int] = (0, -1),
    x_range: tuple[int, int] = (0, -1),
) -> np.ndarray:
    """
    Function that crops an image or a stack of images

    Parameters
    ----------
    data: ndarray
        A 2D or 3D array of images to crop.
    y_range: tuple
        Start and stop y indices for cropping.
    x_range: tuple
        Start and stop x indices for cropping.

    Returns
    -------
    out: ndarray
        Cropped 2D or 3D array, depending on input data
    """

    if data.ndim == 2:
        return np.array(data[y_range[0] : y_range[1], x_range[0] : x_range[1]])
    else:
        return np.array(data[:, y_range[0] : y_range[1], x_range[0] : x_range[1]])


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


def normalise_data(
    data: np.ndarray,
    y_range: tuple[int, int],
    x_range: tuple[int, int],
    dtype: str = "amplitude",
) -> np.ndarray:
    """
    Function that normalises an image or a stack of images with respect to a given ROI.

    Parameters
    ----------
    data: ndarray
        A 2D or 3D array of images to normalise.
    y_range: ndarray
        Start and stop y indices for the ROI.
    x_range: ndarray
        Start and stop x indices for the ROI.
    dtype: str, one of "amplitude" or "phase"
        Data type to be normalised

    Returns
    -------
    normalised: ndarray
        A 2D or 3D array of the normalised images.
    """
    if data.ndim == 2:
        norm_factor = np.mean(data[y_range[0] : y_range[1], x_range[0] : x_range[1]])
        if dtype == "amplitude":
            return data / norm_factor
        elif dtype == "phase":
            return data - norm_factor
        else:
            raise ValueError("dtype must be 'amplitude' or 'phase'")
    else:
        normalised = np.zeros(data.shape)
        for i in range(data.shape[0]):
            norm_factor = np.mean(
                data[i, y_range[0] : y_range[1], x_range[0] : x_range[1]]
            )
            if dtype == "amplitude":
                normalised[i, :, :] = data[i, :, :] / norm_factor
            elif dtype == "phase":
                normalised[i, :, :] = data[i, :, :] - norm_factor
            else:
                raise ValueError("dtype must be 'amplitude' or 'phase'")
        return normalised


def remove_phase_ramp(data: np.ndarray, mask: np.ndarray):
    """
    Function to remove phase ramp

    Parameters
    ----------
    data: ndarray
        Input image as complex 2D array
    mask: ndarray
    """
    if data.ndim == 2:
        return rmphaseramp(data, weight=mask)
    else:
        unramped = np.zeros_like(data)
        for i in range(data.shape[0]):
            unramped[i, :, :] = rmphaseramp(data[i, :, :], weight=mask)
        return unramped
