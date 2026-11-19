from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from ml.explainability import occlusion_explanation, uncertainty_from_scores


EMOTION_LABELS = ["anger", "fear", "joy", "sadness", "disgust", "surprise", "neutral"]


class InferenceEngine:
    def __init__(self) -> None:
        self.status = "unavailable"
        self.mode = "unavailable"
        self.checkpoint: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.labels = EMOTION_LABELS
        project_root = Path(__file__).resolve().parents[1]
        requested = os.getenv("ECHOSAFE_MODEL_CHECKPOINT")
        candidates = [Path(requested)] if requested else [
            project_root / "ml" / "checkpoints" / "echosafe.pt",
            project_root / "backend" / "checkpoints" / "echosafe.pt",
        ]
        for candidate in candidates:
            resolved = Path(candidate)
            if not resolved.is_absolute():
                resolved = project_root / resolved
            if resolved.exists():
                self._load_checkpoint(str(resolved))
                break

    def _load_checkpoint(self, path: str) -> None:
        try:
            import torch
            from ml.model import EchoSafeModel
            from ml.tokenizer import WordTokenizer

            payload = torch.load(path, map_location="cpu", weights_only=True)
            config = payload["model_config"]
            model = EchoSafeModel(
                vocab_size=int(config["vocab_size"]),
                emotion_classes=int(config["emotion_classes"]),
            )
            model.load_state_dict(payload["model_state"])
            model.eval()
            vocab_path = str(Path(path).with_suffix(".vocab.json"))
            if not Path(vocab_path).exists():
                return
            self.model = model
            self.tokenizer = WordTokenizer.load(vocab_path)
            self.labels = list(payload.get("emotion_labels", EMOTION_LABELS))
            self.checkpoint = str(path)
            self.status = "available"
            self.mode = "checkpoint"
        except Exception:
            self.model = None
            self.tokenizer = None
            self.status = "unavailable"
            self.mode = "unavailable"

    def model_info(self) -> Dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "checkpoint": self.checkpoint,
            "emotion_labels": self.labels,
            "emotion_output": "softmax_single_label",
            "architecture": {
                "message_encoder": ["CNN", "BiGRU", "attention pooling"],
                "context_encoder": "masked context LSTM",
                "context_attention": True,
                "heads": ["emotion", "stress"],
            },
        }

    def analyze(self, text: str, context: Optional[List[str]] = None) -> Dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The trained emotion/stress checkpoint is unavailable.")
        return self._checkpoint_analyze(text, context or [])

    def _checkpoint_analyze(self, text: str, context: List[str], explain: bool = True) -> Dict:
        import torch

        messages = list(context[-5:]) + [text]
        encoded = [self.tokenizer.encode(message, max_length=128) for message in messages]
        tensor = torch.tensor([encoded], dtype=torch.long)
        lengths = torch.tensor([len(messages)], dtype=torch.long)
        with torch.no_grad():
            output = self.model(tensor, lengths, return_attention=False)
        emotion_probs = output["emotion_probs"][0].tolist()
        stress = float(output["stress_prob"][0].item())
        emotions = [
            {"label": label, "score": float(prob)}
            for label, prob in zip(self.labels, emotion_probs)
        ]
        emotions.sort(key=lambda item: item["score"], reverse=True)
        uncertainty = uncertainty_from_scores(emotion_probs)
        explanations = occlusion_explanation(
            text,
            lambda value: self._checkpoint_analyze(value, context, explain=False),
        ) if explain else []
        return {
            "emotions": emotions,
            "stress_score": stress,
            "stress_label": "elevated" if stress >= 0.5 else "not elevated",
            "uncertainty": uncertainty,
            "explanations": explanations,
            "model_status": "available",
            "mode": "checkpoint",
            "context": context,
            "analysis_type": "contextual" if context else "individual",
            "disclaimer": "Language-pattern scores, not a diagnosis.",
        }
