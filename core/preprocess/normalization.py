import numpy as np


def normalize_image(image, dtype=np.float32, clip_percentiles=(1.0, 99.0)):
    """Normalize image to [0.0, 1.0] with robust percentile clipping.

    Parameters
    ----------
    image : array-like
        Input image array.
    dtype : np.dtype, optional
        Target float output dtype (default np.float32).
    clip_percentiles : tuple of float or None, optional
        (p_low, p_high) percentiles for contrast clipping (default 1.0 to 99.0).
        If None, standard min-max scaling is performed without clipping.

    Returns
    -------
    normalized : np.ndarray
        Scaled image array with values in [0.0, 1.0].
    mask : np.ndarray
        Boolean validity mask (True for finite pixels).
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]

    img_float = img.astype(np.float32)
    mask = np.isfinite(img_float)

    valid_pixels = img_float[mask]
    if valid_pixels.size == 0:
        norm = np.zeros_like(img_float, dtype=np.float32)
    else:
        if clip_percentiles is not None and len(clip_percentiles) == 2:
            p_low, p_high = float(clip_percentiles[0]), float(clip_percentiles[1])
            low_val, high_val = np.percentile(valid_pixels, [p_low, p_high])
        else:
            low_val = float(valid_pixels.min())
            high_val = float(valid_pixels.max())

        if high_val > low_val:
            clipped = np.clip(img_float, low_val, high_val)
            norm = (clipped - low_val) / (high_val - low_val)
        else:
            minv = float(valid_pixels.min())
            maxv = float(valid_pixels.max())
            if maxv > minv:
                norm = (img_float - minv) / (maxv - minv)
            else:
                norm = np.zeros_like(img_float, dtype=np.float32)

    if norm.shape[-1] == 1:
        norm = norm[..., 0]
    if mask.shape[-1] == 1:
        mask = mask[..., 0]

    return norm.astype(dtype), mask


def mask_invalid(image):
    """Return boolean mask where image values are finite."""
    return np.isfinite(np.asarray(image))
