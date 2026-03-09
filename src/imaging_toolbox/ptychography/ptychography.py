import numpy as np
from ptypy.utils.plot_utils import rmphaseramp  # type: ignore


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
