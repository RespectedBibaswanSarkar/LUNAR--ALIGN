import numpy as np


def normalize_image(image, dtype=np.float32):
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    if img.dtype.kind in {"u", "i"}:
        minv = img.min()
        maxv = img.max()
        if maxv > minv:
            img = (img.astype(np.float32) - minv) / (maxv - minv)
        else:
            img = np.zeros_like(img, dtype=np.float32)
    else:
        img = img.astype(np.float32)
    if img.shape[-1] == 1:
        img = img[..., 0]
    mask = np.isfinite(img)
    return img.astype(dtype), mask


def mask_invalid(image):
    return np.isfinite(np.asarray(image))
