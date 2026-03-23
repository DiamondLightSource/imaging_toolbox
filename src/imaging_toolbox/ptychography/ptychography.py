from typing import cast

import numpy as np
import numpy.typing as npt
from skimage.restoration import unwrap_phase  # type: ignore


def phase_ramp_removal(a: npt.NDArray[np.float64], w: npt.NDArray[np.bool]):
    """
    Function to remove phase ramp

    Parameters
    ----------
    a: ndarray
        2D array containing input phase data
    w: ndarray
        2D array, of equal dimension to a, containing a boolean mask

    Returns
    -------
    out: ndarray
        2D array, of equal dimension to a, containing the corrected phase data with the
        ramp removed
    """
    ph = np.exp(1j * a)
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j * gx / ph)
    gy = -np.real(1j * gy / ph)

    nrm = w.sum()
    agx = (gx * w).sum() / nrm
    agy = (gy * w).sum() / nrm

    (xx, yy) = np.indices(a.shape)
    p = np.exp(-1j * (agx * xx + agy * yy))
    return np.angle(ph * p)


def remove_ramp_and_unwrap_phase(
    data: npt.NDArray[np.float64], mask: npt.NDArray[np.bool]
) -> npt.NDArray[np.float64]:
    """
    Function to remove the ramp and unwrap phase data

    Parameters
    ----------
    data: ndarray
        Phase segment of ptychography data as a 2D or 3D array
    mask: ndarray
        Boolean array that can be 2D or 3D depending on the dimensions of data.
        mask must be 2D if data is 2D and either 2D or 3D if data is 3D

    Returns
    -------
    out: ndarray
        An array with equal dimensions to data containing phase data with ramp removed
        and the phase unwrapped
    """
    if data.ndim == 2:
        if mask.ndim == 3:
            raise ValueError("mask cannot have more dimensions than data")
        if data.shape != mask.shape:
            raise ValueError(
                f"data does not have the same shape as mask.\
                    data has shape {data.shape} whilst mask has shape {mask.shape}"
            )
        return cast(
            npt.NDArray[np.float64],
            unwrap_phase(phase_ramp_removal(data, w=mask)),
        )
    else:
        unramped = np.zeros_like(data)
        if mask.ndim == 2:
            if data.shape[1:] != mask.shape:
                raise ValueError(
                    f"shape of mask {mask.shape} does not \
                                 match the shape of images in data {data.shape}"
                )
            for i in range(data.shape[0]):
                unramped[i, :, :] = unwrap_phase(
                    phase_ramp_removal(data[i, :, :], w=mask)
                )
        else:
            if data.shape != mask.shape:
                raise ValueError(
                    f"shape of mask {mask.shape} does not \
                                 match the shape of data {data.shape}"
                )
            for i in range(data.shape[0]):
                unramped[i, :, :] = unwrap_phase(
                    phase_ramp_removal(data[i, :, :], w=mask[i])
                )
        return unramped
