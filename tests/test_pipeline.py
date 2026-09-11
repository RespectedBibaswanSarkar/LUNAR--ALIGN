import json
import os
import tempfile

import numpy as np

from core.geometry.haversine import haversine_distance
from core.preprocess.normalization import normalize_image
from core.preprocess.pyramid import build_pyramid
from core.registration.pipeline import LunarAlignPipeline
from matchers.sift_baseline.matcher import SIFTMatcher
from core.spatial.grid_selection import select_spatial_matches


def test_haversine_distance():
    d = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert d > 0
    assert np.isfinite(d)


def test_image_normalization_and_pyramid():
    img = np.array([[0, 10], [20, 255]], dtype=np.uint8)
    norm, mask = normalize_image(img)
    assert norm.shape == img.shape
    assert mask.shape == img.shape
    levels = build_pyramid(img, levels=3)
    assert len(levels) == 3


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
