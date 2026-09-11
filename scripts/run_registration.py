#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from core.registration.pipeline import LunarAlignPipeline


def _load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            key = data.files[0]
            return data[key]
        return data
    with Image.open(path) as img:
        return np.asarray(img)


def main():
    parser = argparse.ArgumentParser(description='LUNAR-ALIGN registration CLI')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()

    ref = _load_image(args.reference)
    tgt = _load_image(args.target)
    pipeline = LunarAlignPipeline(output_dir=args.output_dir)
    result = pipeline.run(ref, tgt)

    print("=========================================")
    print("        LUNAR-ALIGN REGISTRATION")
    print("=========================================")
    print(f"RMSE: {result['rmse']:.6f}")
    print(f"Inlier Ratio: {result['inlier_ratio']:.4f}")
    print(f"Spatial Coverage: {result['coverage']:.4f}")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Matcher: {result['matcher']}")
    print("=========================================")


if __name__ == '__main__':
    main()
