"""Run the complete available EchoSafe training workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.train import train
from ml.evaluate import evaluate


def train_all(data_dir: str = "ml/data/processed", epochs: int = 3) -> dict:
    data_path = ROOT / data_dir
    train_fixture = data_path / "train.jsonl"
    validation_fixture = data_path / "validation.jsonl"
    if not train_fixture.exists():
        raise FileNotFoundError(
            f"Training fixture missing: {train_fixture}. Run scripts/prepare_data.py --synthetic first."
        )
    if not validation_fixture.exists():
        raise FileNotFoundError(f"Validation fixture missing: {validation_fixture}")

    combined = data_path / "train_all.jsonl"
    with combined.open("w", encoding="utf-8") as output:
        for source in (train_fixture, validation_fixture):
            with source.open("r", encoding="utf-8") as handle:
                output.writelines(handle)

    checkpoint = ROOT / "ml" / "checkpoints" / "echosafe.pt"
    result = train(str(combined), str(checkpoint), epochs)
    return {
        **result,
        "validation_fixture": str(validation_fixture),
        "metrics": "not_evaluated",
        "note": "No experiment metrics are invented; evaluate a real trained artifact separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EchoSafe from prepared local data")
    parser.add_argument("--data-dir", default="ml/data/processed")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(train_all(args.data_dir, args.epochs), indent=2))


if __name__ == "__main__":
    main()
