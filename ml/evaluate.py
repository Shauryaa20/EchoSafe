from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(checkpoint: str, fixture: str) -> dict:
    if not Path(checkpoint).exists():
        raise FileNotFoundError(checkpoint)
    if not Path(fixture).exists():
        raise FileNotFoundError(fixture)
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # A checkpoint must contain its own evaluator metrics. This module never fabricates them.
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("checkpoint does not contain a metrics dictionary")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EchoSafe checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.fixture), indent=2))


if __name__ == "__main__":
    main()
