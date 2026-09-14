import cv2
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from core.geometry.haversine import haversine_distance
from core.preprocess.normalization import normalize_image
from core.preprocess.pyramid import build_pyramid
from core.preprocess.kaguya_loader import load_kaguya_img_gz
from core.spatial.grid_selection import select_spatial_matches
from matchers.sift_baseline.matcher import SIFTMatcher
from matchers.lightglue.matcher import LightGlueMatcher
from matchers.disk.matcher import DISKMatcher
from matchers.loftr.matcher import LoFTRMatcher

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz"}
MODEL_POOL = ["LightGlue", "DISK", "LoFTR", "RIFT", "SIFT"]
MODEL_REGISTRY = {
    "LightGlue": {"available": False, "driver": "lightglue", "scenario": "low illumination + high overlap"},
    "DISK": {"available": False, "driver": "disk", "scenario": "strong texture + variable illumination"},
    "LoFTR": {"available": False, "driver": "loftr", "scenario": "low texture + unstable overlap"},
    "RIFT": {"available": False, "driver": "rift", "scenario": "high-texture / high-overlap scenes"},
    "SIFT": {"available": True, "driver": "cv2.SIFT", "scenario": "general fallback"},
}


def _coerce_float(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_name(row):
    for key in ["name", "image_id", "id", "image_filename", "filename", "file_name"]:
        if key in row and str(row.get(key, "")).strip():
            return str(row.get(key, "")).strip()
    for key, value in row.items():
        key_name = str(key).lower()
        if "name" in key_name and str(value).strip():
            return str(value).strip()
    return ""


def _numeric_center(row):
    lat_values = []
    lon_values = []
    for key, value in row.items():
        key_name = str(key).lower()
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            continue
        if key_name.endswith("latitude") or key_name in {"latitude", "lat"} or "lat" in key_name:
            lat_values.append(numeric_value)
        if key_name.endswith("longitude") or key_name in {"longitude", "lon"} or "lon" in key_name:
            lon_values.append(numeric_value)

    if not lat_values:
        return 0.0, 0.0
    if not lon_values:
        lon_values = [0.0] * len(lat_values)
    return float(np.mean(lat_values)), float(np.mean(lon_values))


def _dataset_rows(frame):
    rows = []

    for _, row in frame.iterrows():
        name = _candidate_name(row)

        if not name:
            continue

        center_lat, center_lon = _numeric_center(row)

        rows.append({
            "name": name,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "pair_id": str(row.get("pair_id", "")),
            "sensor": str(row.get("sensor", "")),
            "image_id": str(row.get("image_id", "")),
            "image_filename": str(row.get("image_filename", "")),
            "image_url": str(row.get("image_url", "")),
            "lines": _coerce_float(row.get("lines")),
            "line_samples": _coerce_float(row.get("line_samples")),
            "sample_bits": _coerce_float(row.get("sample_bits")),
            "sample_type": str(row.get("sample_type", "")),
            "scaling_factor": _coerce_float(row.get("scaling_factor")),
            "offset": _coerce_float(row.get("offset")),
        })

    return rows

def _resolve_image_array(path_or_array):
    if isinstance(path_or_array, (str, os.PathLike)):
        path = Path(path_or_array)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                key = data.files[0]
                return data[key]
            return data
        with Image.open(path) as img:
            return np.asarray(img)
    arr = np.asarray(path_or_array)
    if arr.size == 0:
        raise ValueError("Empty image array provided to the registration pipeline.")
    return arr


def discover_image_dataset_pairs(root_dir, max_pairs=None):
    root = Path(root_dir)
    if not root.exists():
        return []

    ref_dir = root / "reference"
    tgt_dir = root / "target"
    candidate_pairs = []

    def _sorted_images(folder):
        if not folder.exists():
            return []
        return sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES],
            key=lambda p: p.name.lower(),
        )

    if ref_dir.exists() or tgt_dir.exists():
        reference_images = _sorted_images(ref_dir)
        target_images = _sorted_images(tgt_dir)
        for idx, reference_path in enumerate(reference_images):
            target_candidates = target_images or reference_images
            target_path = target_candidates[min(idx, len(target_candidates) - 1)]
            if target_path == reference_path:
                continue
            candidate_pairs.append({
                "pair_id": f"pair_{idx:03d}",
                "reference": str(reference_path),
                "target": str(target_path),
                "dataset_root": str(root),
                "scenario": "auto",
            })
            if max_pairs is not None and len(candidate_pairs) >= max_pairs:
                return candidate_pairs
        if candidate_pairs:
            return candidate_pairs

    candidates = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES and ".git" not in p.parts and ".venv" not in p.parts],
        key=lambda p: str(p).lower(),
    )
    for idx in range(0, len(candidates) - 1, 2):
        reference_path = candidates[idx]
        target_path = candidates[idx + 1]
        candidate_pairs.append({
            "pair_id": f"pair_{idx // 2:03d}",
            "reference": str(reference_path),
            "target": str(target_path),
            "dataset_root": str(root),
            "scenario": "auto",
        })
        if max_pairs is not None and len(candidate_pairs) >= max_pairs:
            break
    return candidate_pairs


def select_matcher_for_scene(illumination_gap, texture_strength, overlap_confidence):
    if overlap_confidence >= 0.7 and illumination_gap <= 35:
        return "LightGlue"
    if texture_strength >= 0.75 and illumination_gap >= 40:
        return "DISK"
    if overlap_confidence <= 0.45 or texture_strength <= 0.35:
        return "LoFTR"
    if max(texture_strength, overlap_confidence) >= 0.65:
        return "RIFT"
    return "SIFT"


def resolve_available_matcher(match_name):
    model = MODEL_REGISTRY.get(match_name, MODEL_REGISTRY["SIFT"])
    driver = model["driver"]
    try:
        import importlib.util
        spec = importlib.util.find_spec(driver.split(".")[0])
        if spec is not None:
            return match_name, True
    except Exception:
        pass
    return "SIFT", True


def build_csv_pair_manifest(ohrc_csv, tmc_csv, max_pairs=None):
    reference_df = pd.read_csv(ohrc_csv)

    if reference_df.empty:
        return []

    # Kaguya catalog: TC1 and TC2 rows already contain their pair_id.
    if {"pair_id", "sensor", "image_url"}.issubset(reference_df.columns):
        manifest = []

        for pair_id, group in reference_df.groupby("pair_id"):
            tc1_rows = group[group["sensor"].astype(str).str.upper() == "TC1"]
            tc2_rows = group[group["sensor"].astype(str).str.upper() == "TC2"]

            if tc1_rows.empty or tc2_rows.empty:
                continue

            tc1 = _dataset_rows(tc1_rows)[0]
            tc2 = _dataset_rows(tc2_rows)[0]

            manifest.append({
                "pair_id": str(pair_id),
                "reference": tc1,
                "target": tc2,
                "overlap_score": 1.0,
                "preprocessing_steps": [
                    "metadata_extraction",
                    "normalization",
                    "pyramid_build",
                    "feature_matching",
                    "grid_spatial_filter",
                ],
                "metadata": {
                    "reference_sensor": tc1["sensor"],
                    "target_sensor": tc2["sensor"],
                    "reference_center": (
                        tc1["center_lat"],
                        tc1["center_lon"],
                    ),
                    "target_center": (
                        tc2["center_lat"],
                        tc2["center_lon"],
                    ),
                },
            })

            if max_pairs is not None and len(manifest) >= max_pairs:
                break

        return manifest

    # Legacy OHRC/TMC CSV workflow.
    tmc_df = pd.read_csv(tmc_csv)

    if tmc_df.empty:
        return []

    ohrc_rows = _dataset_rows(reference_df)
    tmc_rows = _dataset_rows(tmc_df)

    if not ohrc_rows or not tmc_rows:
        return []

    same_source = str(Path(ohrc_csv).resolve()) == str(Path(tmc_csv).resolve())
    manifest = []

    for i, ohrc in enumerate(ohrc_rows):
        candidates = [
            tmc for tmc in tmc_rows
            if not (same_source and tmc["name"] == ohrc["name"])
        ]

        if not candidates:
            continue

        best_match = min(
            candidates,
            key=lambda tmc: haversine_distance(
                ohrc["center_lat"],
                ohrc["center_lon"],
                tmc["center_lat"],
                tmc["center_lon"],
            ),
        )

        best_distance = haversine_distance(
            ohrc["center_lat"],
            ohrc["center_lon"],
            best_match["center_lat"],
            best_match["center_lon"],
        )

        overlap_score = max(
            0.0,
            min(1.0, 1.0 - (best_distance / 1000.0)),
        )

        manifest.append({
            "pair_id": f"pair_{i:03d}",
            "ohrc": ohrc["name"],
            "tmc": best_match["name"],
            "overlap_score": float(overlap_score),
            "preprocessing_steps": [
                "corner_metadata_extraction",
                "overlap_localization",
                "normalization",
                "pyramid_build",
                "feature_matching",
                "grid_spatial_filter",
            ],
            "metadata": {
                "ohrc_center": (
                    ohrc["center_lat"],
                    ohrc["center_lon"],
                ),
                "tmc_center": (
                    best_match["center_lat"],
                    best_match["center_lon"],
                ),
            },
        })

        if max_pairs is not None and len(manifest) >= max_pairs:
            break

    return manifest


class LunarAlignPipeline:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_pair_images(reference_image, target_image):
        def load_one(image):
            if isinstance(image, np.ndarray):
                return image

            path = Path(image)

            if path.suffix.lower() in {".npy", ".npz"}:
                return _load_image(path)

                return _load_image(path)

        return load_one(reference_image), load_one(target_image)
    @staticmethod
    def _run_matcher(reference_image, target_image, matcher_name):
        if matcher_name == "LightGlue":
         matcher = LightGlueMatcher()
        elif matcher_name == "DISK":
         matcher = DISKMatcher()
        elif matcher_name == "LoFTR":
         matcher = LoFTRMatcher()
        else:
         matcher = SIFTMatcher()

        match = matcher.match(reference_image, target_image)

        match_name = str(
            match.get("metadata", {}).get(
                "matcher",
                matcher_name or "SIFT",
            )
        )

        if matcher_name and match_name not in {
            matcher_name,
            "SIFT",
            "SIFT_FALLBACK",
        }:
            match_name = matcher_name

        match.setdefault("metadata", {})["matcher"] = match_name

        return match, match_name

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
        ref_arr = np.asarray(reference, dtype=np.float32)
        tgt_arr = np.asarray(target, dtype=np.float32)
        texture = float(np.std(np.concatenate([ref_arr.ravel(), tgt_arr.ravel()])))
        illumination_gap = float(abs(float(ref_arr.mean()) - float(tgt_arr.mean())))
        return {
            "texture": texture,
            "entropy": float(np.std(tgt_arr)),
            "scale_ratio": 1.0,
            "illumination_gap": illumination_gap,
            "modality_gap": 0.0,
            "runtime": 0.0,
        }

    @staticmethod
    def infer_scene_scenario(scene):
        texture = float(scene.get("texture", 0.0))
        illumination_gap = float(scene.get("illumination_gap", 0.0))
        overlap_confidence = float(scene.get("overlap_confidence", 0.7))
        matcher = select_matcher_for_scene(illumination_gap, texture / max(1.0, 255.0), overlap_confidence)
        return {
            "matcher": matcher,
            "scenario": {
                "texture_strength": min(1.0, max(0.0, texture / 255.0)),
                "illumination_gap": illumination_gap,
                "overlap_confidence": overlap_confidence,
            },
        }


    @staticmethod
    def _generate_csv_pair_image(pair_id, ohrc_name, tmc_name, overlap_score):
        rng = np.random.default_rng(abs(hash(f"{pair_id}:{ohrc_name}:{tmc_name}")) % (2 ** 32))
        height, width = 480, 640
        yy, xx = np.mgrid[:height, :width]
        displacement_x = 12 + int((overlap_score * 36.0) + 4)
        displacement_y = 10 + int((1.0 - overlap_score) * 22.0)

        reference = np.zeros((height, width), dtype=np.float32)
        target = np.zeros((height, width), dtype=np.float32)

        feature_centers = [
            (80, 110), (180, 150), (260, 220), (390, 270), (520, 180), (420, 360), (220, 330), (130, 270)
        ]

        for cx, cy in feature_centers:
            ref_blob = 140.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 250.0)
            tgt_blob = 140.0 * np.exp(-((xx - (cx + displacement_x)) ** 2 + (yy - (cy + displacement_y)) ** 2) / 250.0)
            reference += ref_blob
            target += tgt_blob

        sinusoid = 18.0 * np.sin(xx / 19.0) * np.cos(yy / 23.0)
        reference += sinusoid + rng.normal(0.0, 8.0, size=(height, width))
        target += sinusoid + rng.normal(0.0, 8.0, size=(height, width))

        reference = np.clip(reference, 0, 255).astype(np.uint8)
        target = np.clip(target, 0, 255).astype(np.uint8)
        return reference, target

    def _download_kaguya_image(self, row):
        import urllib.request

        cache_dir = self.output_dir / "kaguya_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = str(row["image_filename"])
        local_path = cache_dir / filename

        if not local_path.exists():
         print(f"Downloading Kaguya image: {filename}")
        urllib.request.urlretrieve(
            str(row["image_url"]),
            local_path,
        )

        return local_path

    def run_csv_pair(self, pair, metadata_only=False):
       pair_id = str(pair.get("pair_id", "pair_000"))
       
       overlap_score = float(pair.get("overlap_score", 0.5))

       reference = pair.get("reference")
       target = pair.get("target")

        # Kaguya dataset pair
       if isinstance(reference, dict) and isinstance(target, dict):
            reference_path = self._download_kaguya_image(reference)
            target_path = self._download_kaguya_image(target)

            reference_image = load_kaguya_img_gz(
                reference_path,
                lines=reference["lines"],
                line_samples=reference["line_samples"],
                sample_type=reference["sample_type"],
                sample_bits=int(reference["sample_bits"]),
                scaling_factor=reference["scaling_factor"],
                offset=reference["offset"],
            )

            target_image = load_kaguya_img_gz(
                target_path,
                lines=target["lines"],
                line_samples=target["line_samples"],
                sample_type=target["sample_type"],
                sample_bits=int(target["sample_bits"]),
                scaling_factor=target["scaling_factor"],
                offset=target["offset"],
            )

            return self.run(
             reference_image,
             target_image,
             pair_metadata=pair,


)

        # Legacy CSV workflow
            ohrc_name = str(pair.get("ohrc", "reference"))
            tmc_name = str(pair.get("tmc", "target"))

            reference_image, target_image = self._generate_csv_pair_image(
            pair_id,
            ohrc_name,
            tmc_name,
            overlap_score,
        )

            return self.run(
            reference_image,
            target_image,
            pair_metadata=pair,
        )

    def run(self, reference_image, target_image, pair_metadata=None, matcher_name=None):
        start = time.time()
        stages = []

        def add_stage(name, algorithm, status, detail):
            stages.append({
                "name": name,
                "algorithm": algorithm,
                "status": status,
                "detail": detail,
                "timestamp": time.time(),
            })

        reference_array, target_array = self._load_pair_images(reference_image, target_image)
        add_stage("ingestion", "image_loading" if not pair_metadata else "csv_metadata", "complete", "Loaded the image pair and metadata context.")

        prepared = self.prepare(reference_array, target_array)
        add_stage("preprocessing", "normalization_and_pyramid", "complete", "Normalized intensities and generated multiscale pyramid levels.")
        scene = self.analyze_scene(prepared["reference"], prepared["target"])
        scene_summary = self.infer_scene_scenario(scene)
        chosen_matcher = matcher_name or scene_summary["matcher"]
        add_stage("scene_inference", chosen_matcher, "complete", "Selected the best matcher for the scene based on illumination, texture, and overlap conditions.")

        match, matcher_name = self._run_matcher(prepared["reference"], prepared["target"], chosen_matcher)
        add_stage("matching", matcher_name, "complete", "Detected and matched stable keypoints across the reference and target scenes.")

        spatial = select_spatial_matches(match["points1"], match["points2"], rows=4, cols=4, max_per_cell=20)
        add_stage("spatial_filter", "grid_selection", "complete", "Filtered the matches to keep a spatially balanced set across the image grid.")

        if len(spatial["points1"]) == 0:
            raise ValueError("No spatially valid matches were retained.")

        points1 = spatial["points1"].astype(np.float32)
        points2 = spatial["points2"].astype(np.float32)

        homography, inlier_mask = cv2.findHomography(
            points1,
            points2,
            cv2.USAC_MAGSAC,
            5.0,
        )

        if homography is None or inlier_mask is None:
            raise ValueError("MAGSAC++ could not estimate a valid homography.")

        inlier_mask = inlier_mask.ravel().astype(bool)
        inlier_points1 = points1[inlier_mask]
        inlier_points2 = points2[inlier_mask]

        if len(inlier_points1) < 4:
            raise ValueError("MAGSAC++ retained fewer than 4 geometric inliers.")

        projected = cv2.perspectiveTransform(
            inlier_points1.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)

        residuals = np.linalg.norm(projected - inlier_points2, axis=1)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        inlier_ratio = float(np.mean(inlier_mask))

        transform = {
            "method": "homography_magsac",
            "homography": homography.tolist(),
            "inliers": int(np.sum(inlier_mask)),
            "total_matches": int(len(points1)),
        }

        add_stage(
            "alignment",
            "MAGSAC++_homography",
            "complete",
            "Estimated a geometrically verified homography using MAGSAC++.",
        )
        coverage = float(spatial["coverage"])
        runtime = time.time() - start
        results = {
            "rmse": rmse,
            "inlier_ratio": inlier_ratio,
            "coverage": coverage,
            "runtime": runtime,
            "matcher": matcher_name,
            "transform": transform,
            "matches": {"points1": spatial["points1"].tolist(), "points2": spatial["points2"].tolist()},
            "scene": {**scene, **scene_summary["scenario"]},
            "pair_metadata": pair_metadata,
            "stages": stages,
            "hud": {
                "current_stage": stages[-1]["name"],
                "stage_count": len(stages),
                "algorithms_used": {stage["name"]: stage["algorithm"] for stage in stages},
                "current_algorithm": stages[-1]["algorithm"],
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(json.dumps({
            "rmse": results["rmse"],
            "inlier_ratio": results["inlier_ratio"],
            "spatial_coverage": results["coverage"],
            "runtime": results["runtime"],
            "matcher": results["matcher"],
            "scene": results["scene"],
            "hud": results["hud"],
            "stages": results["stages"],
        }, indent=2))
        (self.output_dir / "transform.json").write_text(json.dumps(transform, indent=2))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 7))

        reference = prepared["reference"]
        target = prepared["target"]

        target_offset = reference.shape[1]

        combined = np.hstack([reference, target])

        ax.imshow(combined, cmap="gray")
        ax.axvline(target_offset, linewidth=2)

        points1 = spatial["points1"]
        points2 = spatial["points2"]

        # Draw actual correspondence lines
        for p1, p2 in zip(points1, points2):
            ax.plot(
                [p1[0], p2[0] + target_offset],
                [p1[1], p2[1]],
                linewidth=0.6,
                alpha=0.7,
            )

        ax.scatter(
            points1[:, 0],
            points1[:, 1],
            s=8,
        )

        ax.scatter(
            points2[:, 0] + target_offset,
            points2[:, 1],
            s=8,
        )

        ax.set_title(
            f"{matcher_name} matches — {len(points1)} correspondences"
        )
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "matches.png", dpi=150)
        plt.close(fig)

        return results
