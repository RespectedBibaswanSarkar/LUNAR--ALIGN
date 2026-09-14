import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint


class LightGlueMatcher:
    def __init__(self, max_num_keypoints=2048):
        self.device = torch.device("cpu")

        self.extractor = (
            SuperPoint(max_num_keypoints=max_num_keypoints)
            .eval()
            .to(self.device)
        )

        self.matcher = (
            LightGlue(features="superpoint")
            .eval()
            .to(self.device)
        )

    @staticmethod
    def _to_uint8(image):
        image = np.asarray(image)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)

        low, high = np.percentile(image, (1, 99))

        if high <= low:
            return np.zeros_like(image, dtype=np.uint8)

        image = np.clip(image, low, high)
        image = (image - low) / (high - low) * 255.0

        return image.astype(np.uint8)

    @staticmethod
    def _to_tensor(image):
        tensor = torch.from_numpy(image).float() / 255.0
        return tensor[None, None]

    def match(self, image1, image2):
        gray1 = self._to_uint8(image1)
        gray2 = self._to_uint8(image2)

        tensor1 = self._to_tensor(gray1).to(self.device)
        tensor2 = self._to_tensor(gray2).to(self.device)

        feats1 = self.extractor.extract(tensor1)
        feats2 = self.extractor.extract(tensor2)

        matches_output = self.matcher(
            {
                "image0": feats1,
                "image1": feats2,
            }
        )

        matches = matches_output["matches"][0]

        if matches.shape[0] < 2:
            raise ValueError(
                "LightGlue failed to produce enough matches."
            )

        points1 = (
            feats1["keypoints"][0][matches[:, 0]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        points2 = (
            feats2["keypoints"][0][matches[:, 1]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        confidence = float(
            matches.shape[0] / max(feats1["keypoints"][0].shape[0], 1)
        )

        return {
            "points1": points1,
            "points2": points2,
            "confidence": confidence,
            "metadata": {
                "matcher": "LightGlue",
                "keypoints_reference": int(
                    feats1["keypoints"][0].shape[0]
                ),
                "keypoints_target": int(
                    feats2["keypoints"][0].shape[0]
                ),
                "matches": int(matches.shape[0]),
            },
        }