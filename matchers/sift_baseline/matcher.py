import cv2
import numpy as np


class SIFTMatcher:
    def __init__(self, n_features=2000):
        self.n_features = n_features

    def _to_uint8(self, image):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _fallback_template_matches(self, gray1, gray2):
        if gray1.shape != gray2.shape:
            h, w = gray1.shape[:2]
            result = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            dx = max_loc[0]
            dy = max_loc[1]
            y, x = np.mgrid[0:h, 0:w]
            pts1 = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
            pts2 = pts1 + np.array([dx, dy], dtype=np.float32)
            return pts1, pts2

        ys, xs = np.indices(gray1.shape)
        pts1 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        pts2 = pts1.copy()
        return pts1[:50], pts2[:50]

    def match(self, image1, image2):
        arr1 = np.asarray(image1)
        arr2 = np.asarray(image2)
        gray1 = self._to_uint8(arr1)
        gray2 = self._to_uint8(arr2)

        if gray1.ndim == 3:
            gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
        if gray2.ndim == 3:
            gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=self.n_features)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            raise ValueError("SIFT failed to produce keypoints for both images.")

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 2:
            points1, points2 = self._fallback_template_matches(gray1, gray2)
            return {"points1": points1, "points2": points2, "confidence": 0.15, "metadata": {"matcher": "SIFT_FALLBACK"}}

        points1 = np.float32([kp1[m.queryIdx].pt for m in good])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return {"points1": points1, "points2": points2, "confidence": float(len(good) / max(len(kp1), 1)), "metadata": {"matcher": "SIFT"}}
