import cv2
import numpy as np
import torch
from kornia.feature import LoFTR


class LoFTRMatcher:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = LoFTR(pretrained="outdoor").eval().to(self.device)

    @staticmethod
    def _to_tensor(image):
        image = np.asarray(image)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)

        low, high = np.percentile(image, (1, 99))

        if high <= low:
            image = np.zeros_like(image, dtype=np.float32)
        else:
            image = np.clip((image - low) / (high - low), 0, 1)

        return torch.from_numpy(image).float()[None, None]

    def match(self, image1, image2):
        tensor1 = self._to_tensor(image1).to(self.device)
        tensor2 = self._to_tensor(image2).to(self.device)

        tensor1 = torch.nn.functional.interpolate(
            tensor1,
            size=(584, 512),
            mode="bilinear",
            align_corners=False,
        )

        tensor2 = torch.nn.functional.interpolate(
            tensor2,
            size=(584, 512),
            mode="bilinear",
            align_corners=False,
        )

        output = self.model(
            {
                "image0": tensor1,
                "image1": tensor2,
            }
        )

        points1 = (
            output["keypoints0"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        points2 = (
            output["keypoints1"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if len(points1) < 2:
            raise ValueError("LoFTR failed to produce enough matches.")

        confidence = float(len(points1))

        return {
            "points1": points1,
            "points2": points2,
            "confidence": confidence,
            "metadata": {
                "matcher": "LoFTR",
                "matches": int(len(points1)),
            },
        }