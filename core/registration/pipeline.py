import json
import os
import time
from pathlib import Path

import numpy as np

from core.geometry.haversine import haversine_distance
from core.preprocess.normalization import normalize_image
from core.preprocess.pyramid import build_pyramid
from core.spatial.grid_selection import select_spatial_matches
from matchers.sift_baseline.matcher import SIFTMatcher


class LunarAlignPipeline:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, reference_image, target_image):
        ref_norm, ref_mask = normalize_image(reference_image)
        tgt_norm, tgt_mask = normalize_image(target_image)
        return {
            "reference": ref_norm,
            "target": tgt_norm,
            "reference_mask": ref_mask,
            "target_mask": tgt_mask,
            "pyramid": {"reference": build_pyramid(ref_norm, 3), "target": build_pyramid(tgt_norm, 3)},
        }

    def analyze_scene(self, reference, target):
        return {
            "texture": float(np.std(reference)),
            "entropy": float(np.std(target)),
            "scale_ratio": 1.0,
            "illumination_gap": float(abs(float(reference.mean()) - float(target.mean()))),
            "modality_gap": 0.0,
            "runtime": 0.0,
        }

    def run(self, reference_image, target_image):
        start = time.time()
        prepared = self.prepare(reference_image, target_image)
        scene = self.analyze_scene(prepared["reference"], prepared["target"])

        matcher = SIFTMatcher()
        match = matcher.match(prepared["reference"], prepared["target"])
        spatial = select_spatial_matches(match["points1"], match["points2"], rows=4, cols=4, max_per_cell=20)

        if len(spatial["points1"]) == 0:
            raise ValueError("No spatially valid matches were retained.")

        transform = {
            "translation_x": float(np.mean(spatial["points2"][:, 0] - spatial["points1"][:, 0])),
            "translation_y": float(np.mean(spatial["points2"][:, 1] - spatial["points1"][:, 1])),
            "method": "translation_estimate"
        }

        error = np.linalg.norm((spatial["points2"] - spatial["points1"]), axis=1)
        rmse = float(np.sqrt(np.mean(error ** 2))) if len(error) > 0 else 0.0
        inlier_ratio = 1.0
        results = {
            "rmse": rmse,
            "inlier_ratio": inlier_ratio,
            "coverage": float(spatial["coverage"]),
            "runtime": time.time() - start,
            "matcher": "SIFT",
            "transform": transform,
            "matches": {"points1": spatial["points1"].tolist(), "points2": spatial["points2"].tolist()},
            "scene": scene,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(json.dumps({
            "rmse": results["rmse"],
            "inlier_ratio": results["inlier_ratio"],
            "spatial_coverage": results["coverage"],
            "runtime": results["runtime"],
            "matcher": results["matcher"],
            "scene": results["scene"],
        }, indent=2))
        (self.output_dir / "transform.json").write_text(json.dumps(transform, indent=2))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(prepared["reference"], cmap="gray")
        ax[0].set_title("Reference")
        ax[1].imshow(prepared["target"], cmap="gray")
        ax[1].set_title("Target")
        plt.savefig(self.output_dir / "matches.png", dpi=150)
        plt.close(fig)

        return results
