import cv2 as cv
import numpy as np
from scipy.ndimage import label  # Labeling utility for image segmentation


def white_balance(img):
    """
    Applies white balance to an image by adjusting color channels based on average LAB channel values.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR color space.

    Returns
    -------
    np.ndarray
        Image with applied white balance in BGR color space.

    Raises
    ------
    cv2.error
        If the image cannot be converted to LAB color space.

    Examples
    --------
    >>> img = cv.imread("image.jpg")
    >>> balanced_img = white_balance(img)
    """
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result


def CLAHE(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to each color channel of an image.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR color space.

    Returns
    -------
    np.ndarray
        Image with CLAHE applied to each color channel.

    Raises
    ------
    cv2.error
        If CLAHE cannot be applied to the image channels.

    Examples
    --------
    >>> img = cv.imread("image.jpg")
    >>> enhanced_img = CLAHE(img)
    """
    R, G, B = cv.split(img)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1_R = clahe.apply(R)
    cl1_G = clahe.apply(G)
    cl1_B = clahe.apply(B)
    cl1 = cv.merge((cl1_R, cl1_G, cl1_B))
    return cl1


def normalize(img):
    """
    Applies a series of preprocessing steps, including white balance and CLAHE, to normalize the image.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR color space.

    Returns
    -------
    np.ndarray
        Normalized image with white balance and CLAHE applied.

    Examples
    --------
    >>> img = cv.imread("image.jpg")
    >>> normalized_img = normalize(img)
    """
    img_wb = white_balance(img)
    img_wb_CLAHE = CLAHE(img_wb)
    return img_wb_CLAHE


# post processing functions


def lc_post(mask):
    """
    Performs post-processing on a binary mask to keep only large connected components based on a size threshold.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask image with connected components.

    Returns
    -------
    np.ndarray
        Boolean mask where only large connected components are retained.

    Raises
    ------
    ValueError
        If the input mask is not a 2D array.

    Examples
    --------
    >>> mask = np.zeros((512, 512), dtype=np.uint8)
    >>> mask[100:200, 100:200] = 1  # Example component
    >>> filtered_mask = lc_post(mask)
    """
    size_threshold = 3000
    labeled_mask, num_features = label(mask)
    sizes = np.bincount(labeled_mask.ravel())
    large_component_mask = np.zeros_like(mask, dtype=bool)
    for component_id, size in enumerate(sizes[1:], start=1):
        if size >= size_threshold:
            large_component_mask |= (labeled_mask == component_id)
    return large_component_mask
