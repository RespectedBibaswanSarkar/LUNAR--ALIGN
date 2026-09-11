#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_registration import _load_image
from core.registration.pipeline import LunarAlignPipeline


def _read_manifest(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {path} is empty or missing a header row.")

        required = {"reference", "target"}
        missing = required - set((field.strip() for field in reader.fieldnames))
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required columns: {sorted(missing)}. "
                "Expected at least 'reference' and 'target'."
            )

        rows = []
        for idx, row in enumerate(reader, start=1):
            reference = (row.get("reference") or "").strip()
            target = (row.get("target") or "").strip()
            pair_id = (row.get("pair_id") or f"pair_{idx:04d}").strip()
            if not reference or not target:
                continue
            rows.append({"pair_id": pair_id, "reference": reference, "target": target})
        return rows


def main():
    parser = argparse.ArgumentParser(description="Run lunar registration over a dataset manifest.")
    parser.add_argument("--manifest", required=True, help="CSV file containing reference and target image paths.")
    parser.add_argument("--output-dir", default="outputs/dataset", help="Directory where per-pair outputs are stored.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _read_manifest(manifest_path)
    if not pairs:
        raise ValueError(f"No valid rows were found in {manifest_path}.")

    rows = []
    for pair in pairs:
        pair_dir = output_dir / pair["pair_id"]
        reference = _load_image(str((manifest_path.parent / pair["reference"]).resolve()))
        target = _load_image(str((manifest_path.parent / pair["target"]).resolve()))

        result = LunarAlignPipeline(output_dir=str(pair_dir)).run(reference, target)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "reference": pair["reference"],
                "target": pair["target"],
                "rmse": round(result["rmse"], 6),
                "inlier_ratio": round(result["inlier_ratio"], 6),
                "coverage": round(result["coverage"], 6),
                "runtime": round(result["runtime"], 6),
                "matcher": result["matcher"],
            }
        )

    summary_path = output_dir / "batch_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "reference", "target", "rmse", "inlier_ratio", "coverage", "runtime", "matcher"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} image pairs.")
    print(f"Summary written to {summary_path}")
    for row in rows:
        print(
            f"{row['pair_id']}: RMSE={row['rmse']:.6f}, "
            f"Inlier={row['inlier_ratio']:.4f}, "
            f"Coverage={row['coverage']:.4f}, "
            f"Matcher={row['matcher']}"
        )


if __name__ == "__main__":
    main()
