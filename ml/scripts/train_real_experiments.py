"""Download, prepare, train, and evaluate the official Hugging Face datasets.

The synthetic fixture is intentionally not referenced here.  Run from the
repository root with ``python -m ml.scripts.train_real_experiments``.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ml.real_model import TextClassifier

from ml.tokenizer import WordTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "experiments"
CHECKPOINTS = ROOT / "ml" / "checkpoints" / "real"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def prepare_dataset(name: str, seed: int = 42) -> dict[str, Any]:
    """Download an official HF dataset and write immutable JSONL split copies."""
    from datasets import load_dataset

    if name == "go_emotions":
        ds = load_dataset("google-research-datasets/go_emotions")
        labels = list(ds["train"].features["labels"].feature.names)
        splits = {key: ds[key] for key in ("train", "validation", "test")}
        records = {
            key: [{"text": str(row["text"]), "labels": list(row["labels"])}
                  for row in value]
            for key, value in splits.items()
        }
        task = {"name": name, "dataset_id": "google-research-datasets/go_emotions",
                "labels": labels, "multilabel": True, "splits": list(records)}
    elif name == "dreaddit":
        ds = load_dataset("andreagasparini/dreaddit")
        train = ds["train"]
        test = ds["test"]
        y = [int(x) for x in train["label"]]
        idx_train, idx_val = train_test_split(
            np.arange(len(y)), test_size=0.1, random_state=seed, stratify=y
        )
        def rows(table, indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
            source = range(len(table)) if indices is None else indices
            return [{"text": str(table[int(i)]["text"]), "label": int(table[int(i)]["label"])}
                    for i in source]
        records = {"train": rows(train, idx_train), "validation": rows(train, idx_val),
                   "test": rows(test)}
        labels = ["not_stress", "stress"]
        task = {"name": name, "dataset_id": "andreagasparini/dreaddit",
                "labels": labels, "multilabel": False,
                "splits": list(records), "validation_source": "stratified 10% of official train"}
    else:
        raise ValueError(f"unknown dataset: {name}")
    target = OUT / "data" / name
    target.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    task["counts"] = {key: len(value) for key, value in records.items()}
    task["seed"] = seed
    (target / "manifest.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    return {"task": task, "records": records}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], targets: np.ndarray, tokenizer: WordTokenizer,
                 max_length: int) -> None:
        self.ids = torch.tensor([tokenizer.encode(t, max_length=max_length) for t in texts],
                                dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ids[index], self.targets[index]



def _write_confusion(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _write_per_class(run_dir: Path, metrics: dict[str, Any]) -> None:
    values = metrics.get("per_class") or metrics.get("per_label") or {}
    if not values:
        return
    labels = list(values)
    (run_dir / "per_class.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    fields = ["class", "precision", "recall", "f1", "support"]
    with (run_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            writer.writerow({"class": label, **values[label]})


def _write_confusion_json(run_dir: Path, matrix: np.ndarray, labels: list[str]) -> None:
    payload = {"labels": labels, "matrix": matrix.astype(int).tolist()}
    (run_dir / "confusion_matrix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _plot_history(run_dir: Path) -> None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return
    import matplotlib.pyplot as plt
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [item["train_loss"] for item in history], marker="o", label="train")
    if "validation_loss" in history[0]:
        ax.plot(epochs, [item["validation_loss"] for item in history], marker="o", label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.set_title(run_dir.name)
    ax.legend(); fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=140); plt.close(fig)


def _write_dataset_distribution(name: str, records: dict[str, list[dict[str, Any]]],
                                labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    counts: dict[str, list[int]] = {}
    if name == "go_emotions":
        for split, rows in records.items():
            counts[split] = [sum(label in row["labels"] for row in rows)
                            for label in range(len(labels))]
    else:
        for split, rows in records.items():
            counts[split] = [sum(row["label"] == index for row in rows)
                             for index in range(len(labels))]
    target = OUT / f"{name}_dataset_distribution.png"
    x = np.arange(len(labels)); width = 0.8 / len(counts)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.38), 5))
    for offset, (split, values) in enumerate(counts.items()):
        ax.bar(x + offset * width, values, width, label=split)
    ax.set_xticks(x + width * (len(counts) - 1) / 2, labels, rotation=65, ha="right")
    ax.set_ylabel("Examples"); ax.set_title(f"{name} label distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(target, dpi=140); plt.close(fig)


def _load_test_predictions(name: str, model_name: str,
                           records: dict[str, list[dict[str, Any]]],
                           labels: list[str]) -> np.ndarray:
    """Recreate predictions from saved artifacts, without changing metrics."""
    texts = [row["text"] for row in records["test"]]
    if model_name == "tfidf_logistic_regression":
        bundle = joblib.load(CHECKPOINTS / f"{name}_tfidf.joblib")
        return bundle["classifier"].predict(bundle["vectorizer"].transform(texts))
    run_dir = OUT / f"{name}_{model_name}"
    tokenizer = WordTokenizer.load(run_dir / "vocab.json")
    checkpoint = torch.load(CHECKPOINTS / f"{name}_{model_name}.pt", map_location="cpu")
    model = TextClassifier(len(tokenizer.itos), len(labels) if name == "go_emotions" else 1,
                           model_name)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    ids = torch.tensor([tokenizer.encode(text, max_length=96) for text in texts])
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in DataLoader(ids, batch_size=64):
            predictions.append(torch.sigmoid(model(batch)).numpy())
    values = np.concatenate(predictions)
    return (values >= 0.5).astype(int)


def generate_artifacts(prepared: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Create all derived artifacts from saved metrics/checkpoints."""
    import matplotlib.pyplot as plt
    all_results: list[dict[str, Any]] = []
    for name, bundle in prepared.items():
        task, records = bundle["task"], bundle["records"]
        labels = task["labels"]
        _write_dataset_distribution(name, records, labels)
        rows: list[dict[str, Any]] = []
        metrics_by_model: dict[str, dict[str, Any]] = {}
        for metrics_path in sorted(OUT.glob(f"{name}_*/metrics.json")):
            run_dir = metrics_path.parent
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_name = metrics.get("model", run_dir.name.removeprefix(f"{name}_"))
            _write_per_class(run_dir, metrics)
            _plot_history(run_dir)
            if name == "dreaddit":
                predicted = _load_test_predictions(name, model_name, records, labels)
                actual = np.asarray([row["label"] for row in records["test"]])
                matrix = confusion_matrix(actual, predicted, labels=range(len(labels)))
                _write_confusion_json(run_dir, matrix, labels)
            scalar = {"dataset": name, "model": model_name}
            scalar.update({key: value for key, value in metrics.items()
                           if not isinstance(value, (dict, list))})
            rows.append(scalar); metrics_by_model[model_name] = metrics
            all_results.append(metrics)
        comparison_csv = OUT / f"{name}_model_comparison.csv"
        fields = sorted({key for row in rows for key in row})
        with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        metric = "f1_macro" if name == "go_emotions" else "f1"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([row["model"] for row in rows], [row.get(metric, 0) or 0 for row in rows])
        ax.set_ylabel(metric); ax.set_title(f"{name} model comparison")
        ax.tick_params(axis="x", rotation=35); fig.tight_layout()
        fig.savefig(OUT / f"{name}_model_comparison.png", dpi=140); plt.close(fig)
        if name == "go_emotions":
            per_class_rows = []
            for row in rows:
                metrics = metrics_by_model[row["model"]]
                values = metrics.get("per_label", {})
                for label in labels:
                    per_class_rows.append({"model": row["model"], "class": label,
                                           "f1": values.get(label, {}).get("f1", 0)})
            with (OUT / "go_emotions_per_class_f1.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "class", "f1"])
                writer.writeheader(); writer.writerows(per_class_rows)
            (OUT / "go_emotions_per_class_f1.json").write_text(
                json.dumps(per_class_rows, indent=2), encoding="utf-8")
            fig, ax = plt.subplots(figsize=(12, 6))
            for model in [row["model"] for row in rows]:
                values = [item["f1"] for item in per_class_rows if item["model"] == model]
                ax.plot(labels, values, marker=".", label=model)
            ax.set_ylabel("F1"); ax.set_title("GoEmotions per-class F1")
            ax.tick_params(axis="x", rotation=65); ax.legend()
            fig.tight_layout(); fig.savefig(OUT / "go_emotions_per_class_f1.png", dpi=140); plt.close(fig)
    return all_results


def train_neural(task: dict[str, Any], records: dict[str, list[dict[str, Any]]],
                 architecture: str, epochs: int, batch_size: int, max_length: int,
                 seed: int) -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    multilabel = bool(task["multilabel"])
    labels = task["labels"]; classes = len(labels)
    texts = [row["text"] for row in records["train"]]
    tokenizer = WordTokenizer(vocab_size=20000).fit(texts)
    def target_rows(rows: list[dict[str, Any]]) -> np.ndarray:
        if multilabel:
            result = np.zeros((len(rows), classes), dtype=np.float32)
            for i, row in enumerate(rows):
                result[i, row["labels"]] = 1
            return result
        return np.asarray([[row["label"]] for row in rows], dtype=np.float32)
    train_ds = TextDataset(texts, target_rows(records["train"]), tokenizer, max_length)
    val_ds = TextDataset([r["text"] for r in records["validation"]], target_rows(records["validation"]), tokenizer, max_length)
    test_ds = TextDataset([r["text"] for r in records["test"]], target_rows(records["test"]), tokenizer, max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(len(tokenizer.itos), classes if multilabel else 1, architecture).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float]] = []; started = time.perf_counter()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train(); total = 0.0
        for ids, target in train_loader:
            ids, target = ids.to(device), target.to(device)
            logits = model(ids)
            loss_target = target if multilabel else target
            loss = loss_fn(logits, loss_target)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(ids)
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for ids, target in DataLoader(val_ds, batch_size=batch_size):
                ids, target = ids.to(device), target.to(device)
                val_loss = loss_fn(model(ids), target)
                val_total += val_loss.item() * len(ids)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total / len(train_ds),
            "validation_loss": val_total / len(val_ds),
        })
    model.eval(); predictions: list[np.ndarray] = []; truth: list[np.ndarray] = []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for ids, target in DataLoader(test_ds, batch_size=batch_size):
            predictions.append(torch.sigmoid(model(ids.to(device))).cpu().numpy())
            truth.append(target.numpy())
    inference_seconds = time.perf_counter() - inference_started
    average_inference_ms = 1000 * inference_seconds / len(test_ds)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pred = np.concatenate(predictions); actual = np.concatenate(truth)
    if multilabel:
        binary_pred = (pred >= 0.5).astype(int)
        report = classification_report(actual, binary_pred, target_names=labels,
                                       output_dict=True, zero_division=0)
        per_label = {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            } for label in labels
        }
        roc_auc = average_precision = None
        try:
            roc_auc = float(roc_auc_score(actual, pred, average="macro"))
        except ValueError:
            pass
        try:
            average_precision = float(average_precision_score(actual, pred, average="macro"))
        except ValueError:
            pass
        metrics = {
            "subset_accuracy": float(accuracy_score(actual, binary_pred)),
            "precision_micro": float(precision_score(actual, binary_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(actual, binary_pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(actual, binary_pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(actual, binary_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(actual, binary_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(actual, binary_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(actual, binary_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(actual, binary_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(actual, binary_pred, average="weighted", zero_division=0)),
            "roc_auc_macro": roc_auc,
            "pr_auc_macro": average_precision,
            "per_label": per_label,
        }
        matrix = np.zeros((classes, classes), dtype=int)
    else:
        binary_pred = (pred[:, 0] >= 0.5).astype(int); actual_int = actual[:, 0].astype(int)
        try:
            roc_auc = float(roc_auc_score(actual_int, pred[:, 0]))
            pr_auc = float(average_precision_score(actual_int, pred[:, 0]))
        except ValueError:
            roc_auc = pr_auc = None
        report = classification_report(actual_int, binary_pred, target_names=labels,
                                       output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(actual_int, binary_pred)),
            "precision": float(precision_score(actual_int, binary_pred, zero_division=0)),
            "recall": float(recall_score(actual_int, binary_pred, zero_division=0)),
            "f1": float(f1_score(actual_int, binary_pred, zero_division=0)),
            "precision_macro": float(precision_score(actual_int, binary_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(actual_int, binary_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(actual_int, binary_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(actual_int, binary_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(actual_int, binary_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(actual_int, binary_pred, average="weighted", zero_division=0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "per_class": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                } for label in labels
            },
        }
        matrix = confusion_matrix(actual_int, binary_pred, labels=[0, 1])
    run_id = f"{task['name']}_{architecture}"
    run_dir = OUT / run_id; run_dir.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    tokenizer.save(run_dir / "vocab.json")
    torch.save({"model_state": model.state_dict(), "model_config": {
        "vocab_size": len(tokenizer.itos), "classes": classes, "architecture": architecture}},
        CHECKPOINTS / f"{run_id}.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_confusion(run_dir / "confusion_matrix.png", matrix, labels)
    config = {"dataset": task, "architecture": architecture, "epochs": epochs,
              "batch_size": batch_size, "max_length": max_length, "seed": seed,
              "device": str(device), "hardware": platform.platform(),
              "training_seconds": time.perf_counter() - started,
              "parameter_count": parameter_count,
              "trainable_parameter_count": trainable_parameter_count,
              "average_inference_ms_per_sample": average_inference_ms,
              "context_ablation": "unavailable: these are independent messages, not conversations"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=_jsonable), encoding="utf-8")
    return {"dataset": task["name"], "model": architecture, **metrics,
            "training_seconds": config["training_seconds"], "output": str(run_dir)}


def train_tfidf(task: dict[str, Any], records: dict[str, list[dict[str, Any]]],
                seed: int) -> dict[str, Any]:
    started = time.perf_counter(); multilabel = bool(task["multilabel"])
    train_text = [r["text"] for r in records["train"]]; test_text = [r["text"] for r in records["test"]]
    vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train_text); x_test = vectorizer.transform(test_text)
    if multilabel:
        y_train = np.zeros((len(records["train"]), len(task["labels"])), dtype=int)
        y_test = np.zeros((len(records["test"]), len(task["labels"])), dtype=int)
        for i, row in enumerate(records["train"]): y_train[i, row["labels"]] = 1
        for i, row in enumerate(records["test"]): y_test[i, row["labels"]] = 1
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=seed))
        clf.fit(x_train, y_train)
        inference_started = time.perf_counter(); pred = clf.predict(x_test)
        inference_seconds = time.perf_counter() - inference_started
        try:
            scores = clf.predict_proba(x_test)
            roc_auc = float(roc_auc_score(y_test, scores, average="macro"))
            pr_auc = float(average_precision_score(y_test, scores, average="macro"))
        except (AttributeError, ValueError):
            roc_auc = pr_auc = None
        report = classification_report(y_test, pred, target_names=task["labels"],
                                       output_dict=True, zero_division=0)
        metrics = {
            "subset_accuracy": float(accuracy_score(y_test, pred)),
            "precision_micro": float(precision_score(y_test, pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_test, pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "roc_auc_macro": roc_auc,
            "pr_auc_macro": pr_auc,
            "per_label": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                } for label in task["labels"]
            },
        }
        matrix = confusion_matrix(y_test.argmax(1), pred.argmax(1), labels=range(len(task["labels"])))
    else:
        y_train = [r["label"] for r in records["train"]]; y_test = [r["label"] for r in records["test"]]
        clf = LogisticRegression(max_iter=200, random_state=seed).fit(x_train, y_train)
        inference_started = time.perf_counter(); pred = clf.predict(x_test)
        inference_seconds = time.perf_counter() - inference_started
        scores = clf.predict_proba(x_test)[:, 1]
        report = classification_report(y_test, pred, target_names=task["labels"],
                                       output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "pr_auc": float(average_precision_score(y_test, scores)),
            "per_class": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                } for label in task["labels"]
            },
        }
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    run_dir = OUT / f"{task['name']}_tfidf"; run_dir.mkdir(parents=True, exist_ok=True)
    _write_confusion(run_dir / "confusion_matrix.png", matrix, task["labels"])
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf},
                 CHECKPOINTS / f"{task['name']}_tfidf.joblib")
    average_inference_ms = 1000 * inference_seconds / len(test_text)
    result = {"dataset": task["name"], "model": "tfidf_logistic_regression", **metrics,
              "training_seconds": time.perf_counter() - started, "features": x_train.shape[1],
              "average_inference_ms_per_sample": average_inference_ms,
              "context_ablation": "unavailable: no conversation dataset used"}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"dataset": task, "model": result["model"],
        "hardware": platform.platform(), "seed": seed}, indent=2, default=_jsonable), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["go_emotions", "dreaddit"])
    parser.add_argument("--models", nargs="+", default=["tfidf", "cnn", "bigru", "bilstm", "cnn_bigru_attention"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-only", action="store_true",
                        help="Regenerate reports/plots from existing checkpoints and metrics")
    args = parser.parse_args()
    all_results: list[dict[str, Any]] = []
    prepared_by_name: dict[str, dict[str, Any]] = {}
    for name in args.datasets:
        bundle = prepare_dataset(name, args.seed)
        prepared_by_name[name] = bundle
        task, records = bundle["task"], bundle["records"]
        if args.artifacts_only:
            continue
        for model in args.models:
            try:
                result = train_tfidf(task, records, args.seed) if model == "tfidf" else train_neural(
                    task, records, model, args.epochs, args.batch_size, args.max_length, args.seed)
                all_results.append(result); print(json.dumps(result, default=_jsonable))
            except Exception as exc:
                failure = {"dataset": name, "model": model, "status": "failed", "error": repr(exc)}
                all_results.append(failure); print(json.dumps(failure))
    generated_results = generate_artifacts(prepared_by_name)
    failures = [row for row in all_results if row.get("status") == "failed"]
    all_results = generated_results + failures
    OUT.mkdir(parents=True, exist_ok=True)
    prior_path = OUT / "real_experiment_results.json"
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            combined = {(row.get("dataset"), row.get("model")): row for row in prior}
            combined.update({(row.get("dataset"), row.get("model")): row for row in all_results})
            all_results = [row for key, row in combined.items()
                           if key[0] is not None and key[1] is not None]
        except (OSError, ValueError, TypeError):
            pass
    with (OUT / "real_experiment_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in all_results for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in all_results:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    (OUT / "real_experiment_results.json").write_text(json.dumps(all_results, indent=2, default=_jsonable), encoding="utf-8")


if __name__ == "__main__":
    main()
