from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(fixture: str, output_path: str, epochs: int = 3) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .data import deterministic_split, load_synthetic_fixture
    from .losses import emotion_stress_loss
    from .model import EchoSafeModel
    from .tokenizer import WordTokenizer

    records = load_synthetic_fixture(fixture)
    train_records, val_records = deterministic_split(records)
    tokenizer = WordTokenizer().fit(record["text"] for record in records)
    emotion_labels = sorted({str(record["emotion"]) for record in records})
    emotion_to_id = {label: index for index, label in enumerate(emotion_labels)}
    ids = torch.tensor([tokenizer.encode(record["text"]) for record in train_records], dtype=torch.long).unsqueeze(1)
    targets = torch.tensor([emotion_to_id[str(record["emotion"])] for record in train_records], dtype=torch.long)
    stress = torch.tensor([1 if bool(record["stress"]) else 0 for record in train_records], dtype=torch.float)
    loader = DataLoader(TensorDataset(ids, targets, stress), batch_size=min(32, len(ids)), shuffle=True)
    model = EchoSafeModel(vocab_size=len(tokenizer.itos), emotion_classes=len(emotion_labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for batch_ids, emotion_target, stress_target in loader:
            lengths = torch.ones(batch_ids.size(0), dtype=torch.long)
            optimizer.zero_grad()
            model_output = model(batch_ids, lengths)
            loss = emotion_stress_loss(model_output["emotion_logits"], emotion_target,
                                       model_output["stress_logits"], stress_target)
            loss.backward()
            optimizer.step()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = target.with_suffix(".vocab.json")
    tokenizer.save(tokenizer_path)
    torch.save({
        "model_state": model.state_dict(),
        "emotion_labels": emotion_labels,
        "model_config": {"vocab_size": len(tokenizer.itos), "emotion_classes": len(emotion_labels)},
        "metrics": None,
    }, target)
    return {"checkpoint": str(target), "vocab": str(tokenizer_path), "train_records": len(train_records),
            "validation_records": len(val_records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EchoSafe with a local synthetic or user fixture")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", default="ml/checkpoints/echosafe.pt")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(train(args.fixture, args.output, args.epochs), indent=2))


if __name__ == "__main__":
    main()
