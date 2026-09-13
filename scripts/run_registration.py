#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from PIL import Image

from core.registration.pipeline import LunarAlignPipeline, build_csv_pair_manifest


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


def _load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def main():
    parser = argparse.ArgumentParser(description='LUNAR-ALIGN registration CLI')
    parser.add_argument('--reference')
    parser.add_argument('--target')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_cfg = config.get('dataset', {})
    output_cfg = config.get('output', {})

    if not args.reference:
        args.reference = dataset_cfg.get('reference') or dataset_cfg.get('reference_csv')
    if not args.target:
        args.target = dataset_cfg.get('target') or dataset_cfg.get('target_csv')
    if not args.output_dir:
        args.output_dir = output_cfg.get('dir', 'outputs')

    if args.reference is None or args.target is None:
        parser.error('Either pass --reference and --target or set dataset.reference and dataset.target in the config YAML.')

    reference_path = Path(args.reference)
    target_path = Path(args.target)
    if reference_path.suffix.lower() == '.csv' and target_path.suffix.lower() == '.csv':
        manifest = build_csv_pair_manifest(
            reference_path,
            target_path,
            max_pairs=int(dataset_cfg.get('max_pairs', 6)),
        )
        if not manifest:
            raise ValueError(f'No valid pairs were found in {reference_path} and {target_path}.')
        for pair in manifest:
            pair_dir = Path(args.output_dir) / pair['pair_id']
            pipeline = LunarAlignPipeline(output_dir=str(pair_dir))
            pipeline.run_csv_pair(pair, metadata_only=False)
        print(f'Processed {len(manifest)} pairs from the dataset manifest.')
        return

    ref = _load_image(str(reference_path))
    tgt = _load_image(str(target_path))
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
