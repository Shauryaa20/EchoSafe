"""Prepare local EchoSafe fixtures or validate an existing processed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data import deterministic_split, load_synthetic_fixture


def prepare(synthetic: bool = False, data_dir: str = "ml/data") -> dict:
    destination = ROOT / data_dir
    processed = destination / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    if synthetic:
        source = ROOT / "ml" / "fixtures" / "synthetic.jsonl"
        records = load_synthetic_fixture(source)
        train_records, validation_records = deterministic_split(records, train_ratio=0.75, seed=42)

        train_path = processed / "train.jsonl"
        validation_path = processed / "validation.jsonl"
        for path, rows in ((train_path, train_records), (validation_path, validation_records)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {
            "source": "synthetic_fixture",
            "train": str(train_path),
            "validation": str(validation_path),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
        }

    if not processed.is_dir():
        raise FileNotFoundError(f"processed data directory does not exist: {processed}")
    train_path = processed / "train.jsonl"
    validation_path = processed / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Prepared data requires processed/train.jsonl and processed/validation.jsonl"
        )
    return {
        "train": str(train_path),
        "validation": str(validation_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EchoSafe training data")
    parser.add_argument("--synthetic", action="store_true", help="Create deterministic split from bundled fixture")
    parser.add_argument("--data-dir", default="ml/data", help="Data directory relative to project root")
    args = parser.parse_args()
    print(json.dumps(prepare(args.synthetic, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
