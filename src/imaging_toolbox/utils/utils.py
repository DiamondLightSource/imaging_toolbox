import numpy as np


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
