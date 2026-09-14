import gzip
from pathlib import Path

import numpy as np


def load_kaguya_img_gz(
    path,
    lines,
    line_samples,
    sample_type="MSB_INTEGER",
    sample_bits=16,
    scaling_factor=1.0,
    offset=0.0,
):
    """Load a Kaguya PDS3 .img.gz file into a 2-D NumPy array."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Kaguya image not found: {path}")

    if sample_bits != 16:
        raise ValueError(f"Unsupported sample_bits={sample_bits}; expected 16.")

    if sample_type != "MSB_INTEGER":
        raise ValueError(
            f"Unsupported sample_type={sample_type}; expected MSB_INTEGER."
        )

    with gzip.open(path, "rb") as handle:
        raw = handle.read()

    expected_pixels = int(lines) * int(line_samples)
    expected_bytes = expected_pixels * 2

    if len(raw) != expected_bytes:
        raise ValueError(
            f"Unexpected file size: got {len(raw)} bytes, "
            f"expected {expected_bytes} bytes."
        )

    image = np.frombuffer(raw, dtype=">u2").reshape(
        (int(lines), int(line_samples))
    )

    image = image.astype(np.float32)
    image = image * float(scaling_factor) + float(offset)

    return image