import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def _apply_anti_aliasing(img, sigma=1.0):
    """Apply a low-pass Gaussian smoothing filter before decimation."""
    if img.ndim < 2 or img.shape[0] < 3 or img.shape[1] < 3:
        return img

    if cv2 is not None:
        ksize = int(2 * round(2 * sigma) + 1)
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    elif gaussian_filter is not None:
        if img.ndim == 3:
            filtered = np.empty_like(img)
            for c in range(img.shape[2]):
                filtered[..., c] = gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
            return filtered.astype(img.dtype)
        return gaussian_filter(img.astype(np.float32), sigma=sigma).astype(img.dtype)
    else:
        # 3x3 binomial smoothing filter fallback using NumPy
        padded = np.pad(img.astype(np.float32), 1, mode="edge")
        smooth = (
            padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:] +
            2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:] +
            padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0
        return smooth.astype(img.dtype)


def build_pyramid(image, levels=3, sigma=1.0):
    """Build multi-scale image pyramid with anti-aliasing filtering before decimation.

    Parameters
    ----------
    image : array-like
        Input image.
    levels : int, optional
        Number of pyramid levels (default 3).
    sigma : float, optional
        Standard deviation for Gaussian anti-aliasing filter before 2x subsampling.

    Returns
    -------
    list of np.ndarray
        Pyramid levels from finest (original) to coarsest.
    """
    pyramid = []
    current = np.asarray(image)
    for _ in range(levels):
        pyramid.append(current)
        if current.shape[0] <= 2 or current.shape[1] <= 2:
            if len(pyramid) < levels:
                current = current.copy()
            else:
                break
        else:
            filtered = _apply_anti_aliasing(current, sigma=sigma)
            current = filtered[::2, ::2]
    return pyramid
