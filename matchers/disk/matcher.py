import cv2
import numpy as np
import torch
from kornia.feature import DISK, match_mnn


class DISKMatcher:
    def __init__(self, max_num_keypoints=1024):
        self.device = torch.device("cpu")
        self.model = DISK.from_pretrained("depth").eval().to(self.device)
        self.max_num_keypoints = max_num_keypoints

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

        tensor = torch.from_numpy(image).float()

        # DISK expects 3 channels.
        return tensor[None, None].repeat(1, 3, 1, 1)

    def match(self, image1, image2):
        tensor1 = self._to_tensor(image1).to(self.device)
        tensor2 = self._to_tensor(image2).to(self.device)

        original_h1, original_w1 = tensor1.shape[-2:]
        original_h2, original_w2 = tensor2.shape[-2:]

        resized_h, resized_w = 1168, 1024

        tensor1 = torch.nn.functional.interpolate(
            tensor1, size=(resized_h, resized_w),
            mode="bilinear", align_corners=False
        )
        tensor2 = torch.nn.functional.interpolate(
            tensor2, size=(resized_h, resized_w),
            mode="bilinear", align_corners=False
        )

        features1 = self.model(tensor1, n=self.max_num_keypoints)[0]
        features2 = self.model(tensor2, n=self.max_num_keypoints)[0]

        distances, indices = match_mnn(
            features1.descriptors,
            features2.descriptors,
        )

        if indices.shape[0] < 2:
            raise ValueError("DISK failed to produce enough matches.")

        points1 = (
            features1.keypoints[indices[:, 0]]
            .detach().cpu().numpy().astype(np.float32)
        )
        points2 = (
            features2.keypoints[indices[:, 1]]
            .detach().cpu().numpy().astype(np.float32)
        )

        # Convert DISK's resized coordinates back to original-image coordinates.
        points1[:, 0] *= original_w1 / resized_w
        points1[:, 1] *= original_h1 / resized_h

        points2[:, 0] *= original_w2 / resized_w
        points2[:, 1] *= original_h2 / resized_h

        confidence = float(
            indices.shape[0] / max(features1.keypoints.shape[0], 1)
        )

        return {
            "points1": points1,
            "points2": points2,
            "confidence": confidence,
            "metadata": {
                "matcher": "DISK",
                "keypoints_reference": int(features1.keypoints.shape[0]),
                "keypoints_target": int(features2.keypoints.shape[0]),
                "matches": int(indices.shape[0]),
            },
        }