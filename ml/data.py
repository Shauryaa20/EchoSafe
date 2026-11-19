from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_jsonl(path: str | Path) -> Iterable[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_synthetic_fixture(path: str | Path) -> List[Dict]:
    records = list(load_jsonl(path))
    required = {"text", "emotion", "stress"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"fixture record missing fields: {sorted(missing)}")
    return records


def deterministic_split(records: Sequence[Dict], train_ratio: float = 0.8,
                       seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cutoff = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:cutoff], shuffled[cutoff:]
