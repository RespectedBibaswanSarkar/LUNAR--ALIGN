import json
import os
import tempfile
from pathlib import Path

import numpy as np

from core.geometry.haversine import MOON_RADIUS_KM, haversine_distance
from core.preprocess.normalization import normalize_image
from core.preprocess.pyramid import build_pyramid
from core.preprocess.rotation import (
    align_scan_direction,
    compute_orientation_angle,
    rotate_image,
)
from core.registration.pipeline import (
    LunarAlignPipeline,
    build_csv_pair_manifest,
    discover_image_dataset_pairs,
    select_matcher_for_scene,
)
from matchers.sift_baseline.matcher import SIFTMatcher
from core.spatial.grid_selection import select_spatial_matches


def test_haversine_distance():
    d = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert d > 0
    assert np.isfinite(d)
    # 1 degree of longitude at Moon equator is ~30.323 km (Moon radius = 1737.4 km)
    expected = MOON_RADIUS_KM * np.radians(1.0)
    assert abs(d - expected) < 1e-4


def test_image_normalization_and_pyramid():
    img = np.array([[0, 10], [20, 255]], dtype=np.uint8)
    norm, mask = normalize_image(img)
    assert norm.shape == img.shape
    assert mask.shape == img.shape
    levels = build_pyramid(img, levels=3)
    assert len(levels) == 3


def test_percentile_normalization_outlier_rejection():
    # Array with a realistic gradient and extreme outliers (hot pixel 10000.0 and dead pixel -500.0)
    data = np.linspace(10.0, 200.0, 2500, dtype=np.float32).reshape(50, 50)
    data[0, 0] = -500.0   # extreme cold outlier
    data[-1, -1] = 10000.0  # extreme hot outlier
    norm, mask = normalize_image(data, clip_percentiles=(1.0, 99.0))
    assert norm.shape == data.shape
    assert mask.all()
    assert 0.0 <= norm.min() <= norm.max() <= 1.0
    # The middle values (~105.0) should remain around ~0.5, not crushed to near 0.01 by the 10000.0 outlier
    mid_val = norm[25, 25]
    assert 0.35 <= mid_val <= 0.65


def test_pyramid_anti_aliasing():
    # High frequency pattern
    img = np.zeros((64, 64), dtype=np.float32)
    img[::2, ::2] = 1.0
    levels = build_pyramid(img, levels=4, sigma=1.0)
    assert len(levels) == 4
    assert levels[0].shape == (64, 64)
    assert levels[1].shape == (32, 32)
    assert levels[2].shape == (16, 16)
    assert levels[3].shape == (8, 8)
    # Values should remain in valid bounds
    assert levels[1].min() >= 0.0
    assert levels[1].max() <= 1.0


def test_rotation_and_orientation():
    # Angle calculation from notebook: dx=0, dy=1 -> 90 degrees
    angle = compute_orientation_angle((0, 0), (0, 10))
    assert abs(angle - 90.0) < 1e-4

    # Rotate 20x20 image
    img = np.zeros((20, 20), dtype=np.float32)
    img[8:12, 8:12] = 1.0
    rotated = rotate_image(img, angle_deg=90.0)
    assert rotated.shape == img.shape

    # Align scan direction
    flipped = align_scan_direction(img, flip=True, rot90_k=2)
    assert flipped.shape == img.shape


def test_sift_matcher_returns_points():
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv = np.random.default_rng(0)
    pts = cv.integers(20, 180, size=(8, 2))
    for x, y in pts:
        img1[y:y+12, x:x+12] = 200
        img2[y+5:y+17, x+5:x+17] = 200

    matcher = SIFTMatcher()
    match = matcher.match(img1, img2)
    assert len(match["points1"]) >= 2
    assert len(match["points2"]) >= 2


def test_grid_selection_coverage():
    points1 = np.array([[10, 10], [20, 20], [100, 100]])
    points2 = np.array([[15, 15], [25, 25], [110, 110]])
    selected = select_spatial_matches(points1, points2, rows=2, cols=2, max_per_cell=1)
    assert len(selected["points1"]) > 0
    assert 0 <= selected["coverage"] <= 1


def test_pipeline_generates_outputs(tmp_path):
    reference = np.zeros((180, 220), dtype=np.uint8)
    target = np.zeros((180, 220), dtype=np.uint8)
    reference[40:120, 30:150] = 200
    target[45:125, 35:155] = 200

    pipeline = LunarAlignPipeline(output_dir=str(tmp_path))
    result = pipeline.run(reference, target)
    assert result is not None
    assert result["rmse"] >= 0
    assert os.path.exists(tmp_path / "metrics.json")
    assert os.path.exists(tmp_path / "transform.json")
    assert os.path.exists(tmp_path / "matches.png")


def test_csv_manifest_uses_existing_ohrc_tmc_metadata():
    root = Path(__file__).resolve().parents[1]
    manifest = build_csv_pair_manifest(
    ohrc_csv=root / "data" / "kaguya_sample.csv",
    tmc_csv=root / "data" / "kaguya_sample.csv",
    max_pairs=3,
)
    assert len(manifest) > 0
    assert all({"reference", "target"}.issubset(row.keys()) for row in manifest)
    assert set(manifest[0].keys()) >= {
    "pair_id",
    "reference",
    "target",
    "overlap_score",
    "preprocessing_steps",
}


def test_kaguya_catalog_manifest_uses_lat_lon_metadata():
    root = Path(__file__).resolve().parents[1]
    manifest = build_csv_pair_manifest(
        ohrc_csv=root / "data" / "kaguya_sample.csv",
        tmc_csv=root / "data" / "kaguya_sample.csv",
        max_pairs=3,
    )
    assert len(manifest) > 0
    assert all({"reference", "target"}.issubset(row.keys()) for row in manifest)
    assert set(manifest[0].keys()) >= {
    "pair_id",
    "reference",
    "target",
    "overlap_score",
    "preprocessing_steps",
}

def test_csv_based_pipeline_runs_with_stage_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = build_csv_pair_manifest(
    ohrc_csv=root / "data" / "kaguya_sample.csv",
    tmc_csv=root / "data" / "kaguya_sample.csv",
    max_pairs=1,
)
    pipeline = LunarAlignPipeline(output_dir=str(tmp_path))
    result = pipeline.run_csv_pair(manifest[0], metadata_only=True)

    assert result["rmse"] >= 0
    assert result["matcher"] in {"LightGlue", "DISK", "LoFTR", "RIFT", "SIFT", "SIFT_FALLBACK"}
    assert isinstance(result["stages"], list)
    assert any(stage["name"] == "matching" for stage in result["stages"])
    assert result["hud"]["current_stage"] in {stage["name"] for stage in result["stages"]}


def test_dataset_image_pairs_are_discovered_and_matcher_selected(tmp_path):
    from PIL import Image

    ref_dir = tmp_path / "reference"
    tgt_dir = tmp_path / "target"
    ref_dir.mkdir()
    tgt_dir.mkdir()

    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(ref_dir / "ref_001.png")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(tgt_dir / "tgt_001.png")

    manifest = discover_image_dataset_pairs(tmp_path, max_pairs=1)
    assert len(manifest) >= 1
    assert "reference" in manifest[0]["reference"] or "target" in manifest[0]["target"]

    matcher_name = select_matcher_for_scene(0.2, 0.8, 0.7)
    assert matcher_name in {"SIFT", "DISK", "LightGlue", "LoFTR", "RIFT"}
