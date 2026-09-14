import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def compute_orientation_angle(pt1, pt2):
    """Calculate orientation angle (in degrees) of a baseline with the horizontal.

    Matches the logic in ohrc_tmc_preprocessing.ipynb:
    myradians = math.atan2(dx, dy)
    mydegrees = 90.0 - math.degrees(myradians)

    Parameters
    ----------
    pt1 : tuple or array-like of float
        First point (x, y) or (col, row).
    pt2 : tuple or array-like of float
        Second point (x, y) or (col, row).

    Returns
    -------
    angle_deg : float
        Angle in degrees relative to horizontal.
    """
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    radians = math.atan2(dx, dy)
    degrees = 90.0 - math.degrees(radians)
    return float(degrees)


def rotate_image(image, angle_deg, center=None, border_value=0.0):
    """Rotate an image by a given angle in degrees around a center.

    Parameters
    ----------
    image : np.ndarray
        Input 2D or 3D image.
    angle_deg : float
        Rotation angle in degrees.
    center : tuple of float or None, optional
        Center (x, y) of rotation. Defaults to image center.
    border_value : float or int, optional
        Pixel value used for border padding.

    Returns
    -------
    rotated : np.ndarray
        Rotated image with the same dimensions.
    """
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)

    if cv2 is not None:
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            arr,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    # Fallback to scipy if cv2 is not available
    try:
        from scipy.ndimage import rotate
        return rotate(arr, angle_deg, reshape=False, cval=border_value)
    except ImportError:
        return arr.copy()


def align_scan_direction(image, flip=False, rot90_k=0):
    """Re-orient image to standardize orbital scan / flight direction.

    Parameters
    ----------
    image : array-like
        Input image array.
    flip : bool, optional
        If True, applies horizontal flip (np.fliplr).
    rot90_k : int, optional
        Number of times to rotate by 90 degrees (np.rot90).

    Returns
    -------
    aligned : np.ndarray
        Re-oriented image array.
    """
    arr = np.asarray(image)
    if rot90_k != 0:
        arr = np.rot90(arr, k=rot90_k)
    if flip:
        arr = np.fliplr(arr)
    return arr.copy()
