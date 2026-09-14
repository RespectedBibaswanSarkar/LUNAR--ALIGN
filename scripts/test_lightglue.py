import gzip
from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint


REFERENCE = Path(
    "outputs/TC1W2B0_01_00874N004E1572__TC2W2B0_01_00874N004E1572/"
    "kaguya_cache/TC1W2B0_01_00874N004E1572.img.gz"
)

TARGET = Path(
    "outputs/TC1W2B0_01_00874N004E1572__TC2W2B0_01_00874N004E1572/"
    "kaguya_cache/TC2W2B0_01_00874N004E1572.img.gz"
)


def load_kaguya(path, lines, samples):
    with gzip.open(path, "rb") as f:
        raw = f.read()

    image = np.frombuffer(raw, dtype=">u2").reshape(lines, samples)
    image = image.astype(np.float32)

    # Normalize to 0-255 for feature extraction.
    low, high = np.percentile(image, (1, 99))

    image = np.clip(image, low, high)
    image = (image - low) / (high - low) * 255.0

    return image.astype(np.uint8)


def to_tensor(image):
    tensor = torch.from_numpy(image).float() / 255.0
    return tensor[None, None]


print("Loading Kaguya images...")

# These dimensions come from the Kaguya catalog metadata.
reference = load_kaguya(REFERENCE, 4656, 3496)
target = load_kaguya(TARGET, 4656, 3496)
device = torch.device("cpu")

print("Device:", device)
print("Reference:", reference.shape)
print("Target:", target.shape)

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

image0 = {"image": to_tensor(reference).to(device)}
image1 = {"image": to_tensor(target).to(device)}

print("Extracting features...")

feats0 = extractor.extract(image0["image"])
feats1 = extractor.extract(image1["image"])

print("Matching with LightGlue...")

matches01 = matcher({"image0": feats0, "image1": feats1})

matches = matches01["matches"][0]

print("LightGlue test completed.")
print("Keypoints reference:", feats0["keypoints"][0].shape[0])
print("Keypoints target:", feats1["keypoints"][0].shape[0])
print("Matches:", matches.shape[0])