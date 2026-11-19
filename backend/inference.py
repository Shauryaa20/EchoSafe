from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ml.real_model import TextClassifier
from ml.tokenizer import WordTokenizer


GO_EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
]


class GoEmotionInferenceEngine:
    def __init__(self, checkpoint: Path, vocab: Path) -> None:
        self.checkpoint = checkpoint
        self.vocab_path = vocab
        self.status = "unavailable"
        self.error: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.labels = GO_EMOTION_LABELS
        self.parameter_count = 0
        self._load()

    def _load(self) -> None:
        try:
            if not self.checkpoint.exists():
                raise FileNotFoundError(f"GoEmotions checkpoint missing: {self.checkpoint}")
            if not self.vocab_path.exists():
                raise FileNotFoundError(f"GoEmotions vocabulary missing: {self.vocab_path}")
            payload = torch.load(
                str(self.checkpoint), map_location="cpu", weights_only=True
            )
            config = payload["model_config"]
            if int(config["classes"]) != 28:
                raise ValueError("GoEmotions checkpoint output dimension is not 28")
            model = TextClassifier(
                vocab=int(config["vocab_size"]),
                classes=28,
                architecture="cnn_bigru_attention",
            )
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            tokenizer = WordTokenizer.load(str(self.vocab_path))
            if len(tokenizer.itos) != int(config["vocab_size"]):
                raise ValueError(
                    "GoEmotions vocabulary length does not match checkpoint embedding"
                )
            self.model = model
            self.tokenizer = tokenizer
            self.parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            self.status = "available"
        except Exception as exc:
            self.status = "unavailable"
            self.error = f"{type(exc).__name__}: {exc}"

    def predict(self, text: str) -> List[Dict[str, object]]:
        if self.status != "available":
            raise RuntimeError(f"GoEmotions model unavailable: {self.error}")
        ids = torch.tensor(
            [self.tokenizer.encode(text, max_length=96, add_special=True)],
            dtype=torch.long,
        )
        with torch.inference_mode():
            logits = self.model(ids)
            probabilities = torch.sigmoid(logits)[0].tolist()
        scores = [
            {"label": label, "score": float(score)}
            for label, score in zip(self.labels, probabilities)
        ]
        return sorted(scores, key=lambda item: item["score"], reverse=True)

    def model_info(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "error": self.error,
            "checkpoint": str(self.checkpoint),
            "vocabulary": str(self.vocab_path),
            "architecture": "CNN + BiGRU + Attention",
            "labels": self.labels,
            "output_dim": 28,
            "vocabulary_size": len(self.tokenizer.itos) if self.tokenizer else None,
            "parameter_count": self.parameter_count,
            "training_samples": 43410,
            "validation_samples": 5426,
            "test_samples": 5427,
        }


class DreadditStressInferenceEngine:
    def __init__(self, checkpoint: Path, vocab: Path) -> None:
        self.checkpoint = checkpoint
        self.vocab_path = vocab
        self.status = "unavailable"
        self.error: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.parameter_count = 0
        self.threshold = 0.5
        self._load()

    def _load(self) -> None:
        try:
            if not self.checkpoint.exists():
                raise FileNotFoundError(f"Dreaddit checkpoint missing: {self.checkpoint}")
            if not self.vocab_path.exists():
                raise FileNotFoundError(f"Dreaddit vocabulary missing: {self.vocab_path}")
            payload = torch.load(
                str(self.checkpoint), map_location="cpu", weights_only=True
            )
            config = payload["model_config"]
            model = TextClassifier(
                vocab=int(config["vocab_size"]),
                classes=1,
                architecture="cnn_bigru_attention",
            )
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            tokenizer = WordTokenizer.load(str(self.vocab_path))
            embedding_rows = int(model.embedding.weight.shape[0])
            if len(tokenizer.itos) != embedding_rows:
                raise ValueError(
                    "Dreaddit vocabulary length does not match checkpoint embedding"
                )
            self.model = model
            self.tokenizer = tokenizer
            self.parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            self.status = "available"
        except Exception as exc:
            self.status = "unavailable"
            self.error = f"{type(exc).__name__}: {exc}"

    def predict(self, text: str) -> float:
        if self.status != "available":
            raise RuntimeError(f"Dreaddit model unavailable: {self.error}")
        ids = torch.tensor(
            [self.tokenizer.encode(text, max_length=96, add_special=True)],
            dtype=torch.long,
        )
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(ids))[0, 0].item()
        return float(probability)

    def model_info(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "error": self.error,
            "checkpoint": str(self.checkpoint),
            "vocabulary": str(self.vocab_path),
            "architecture": "CNN + BiGRU + Attention",
            "labels": ["not_stress", "stress"],
            "output_dim": 1,
            "threshold": self.threshold,
            "vocabulary_size": len(self.tokenizer.itos) if self.tokenizer else None,
            "parameter_count": self.parameter_count,
            "training_samples": 2554,
            "validation_samples": 284,
            "test_samples": 715,
        }


class RealInferenceAdapter:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        go_checkpoint = Path(os.getenv(
            "ECHOSAFE_GO_EMOTION_CHECKPOINT",
            str(project_root / "ml" / "checkpoints" / "real" / "go_emotions_cnn_bigru_attention.pt"),
        ))
        go_vocab = Path(os.getenv(
            "ECHOSAFE_GO_EMOTION_VOCAB",
            str(project_root / "ml" / "experiments" / "go_emotions_cnn_bigru_attention" / "vocab.json"),
        ))
        stress_checkpoint = Path(os.getenv(
            "ECHOSAFE_DREADDIT_CHECKPOINT",
            str(project_root / "ml" / "checkpoints" / "real" / "dreaddit_cnn_bigru_attention.pt"),
        ))
        stress_vocab = Path(os.getenv(
            "ECHOSAFE_DREADDIT_VOCAB",
            str(project_root / "ml" / "experiments" / "dreaddit_cnn_bigru_attention" / "vocab.json"),
        ))
        self.go_emotions = GoEmotionInferenceEngine(go_checkpoint, go_vocab)
        self.dreaddit = DreadditStressInferenceEngine(stress_checkpoint, stress_vocab)
        self.status = (
            "available"
            if self.go_emotions.status == "available"
            and self.dreaddit.status == "available"
            else "unavailable"
        )
        self.mode = "checkpoint" if self.status == "available" else "unavailable"
        self.checkpoint = None

    def analyze(self, text: str, context: Optional[List[str]] = None) -> Dict[str, object]:
        if self.status != "available":
            errors = []
            if self.go_emotions.error:
                errors.append(f"GoEmotions: {self.go_emotions.error}")
            if self.dreaddit.error:
                errors.append(f"Dreaddit: {self.dreaddit.error}")
            raise RuntimeError("; ".join(errors) or "Real trained models unavailable")

        clean = text.strip()
        emotion_scores = self.go_emotions.predict(clean)
        stress_score = self.dreaddit.predict(clean)
        top_score = float(emotion_scores[0]["score"])
        confidence = max(0.0, min(1.0, top_score))
        is_uncertain = confidence < 0.65

        explanations = self._occlusion_explanations(
            clean, emotion_scores, stress_score
        )
        return {
            "emotions": emotion_scores,
            "stress_score": stress_score,
            "stress_label": "elevated" if stress_score >= 0.5 else "not elevated",
            "uncertainty": {
                "confidence": round(confidence, 6),
                "is_uncertain": is_uncertain,
                "method": "top_goemotions_sigmoid",
            },
            "explanations": explanations,
            "model_status": "available",
            "mode": "checkpoint",
            "context": list(context or []),
            "analysis_type": "individual",
            "disclaimer": "Language-pattern scores, not a diagnosis.",
        }

    def _occlusion_explanations(
        self,
        text: str,
        baseline_emotions: List[Dict[str, object]],
        baseline_stress: float,
    ) -> List[Dict[str, object]]:
        words = text.split()
        base_emotion = float(baseline_emotions[0]["score"])
        base_value = base_emotion + baseline_stress
        explanations: List[Dict[str, object]] = []
        for index, word in enumerate(words[:12]):
            occluded = " ".join(words[:index] + words[index + 1:])
            if not occluded:
                continue
            candidate_emotions = self.go_emotions.predict(occluded)
            candidate_stress = self.dreaddit.predict(occluded)
            candidate_value = (
                float(candidate_emotions[0]["score"]) + candidate_stress
            )
            explanations.append({
                "text": word,
                "importance": round(base_value - candidate_value, 6),
            })
        return sorted(
            explanations,
            key=lambda item: abs(float(item["importance"])),
            reverse=True,
        )

    def model_info(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "checkpoint": None,
            "go_emotions": self.go_emotions.model_info(),
            "dreaddit": self.dreaddit.model_info(),
            "architecture": {
                "go_emotions": "CNN + BiGRU + Attention",
                "dreaddit": "CNN + BiGRU + Attention",
                "context_encoder": False,
                "max_sequence_length": 96,
            },
            "real_trained_checkpoints": True,
        }


# Backward-compatible name used by the existing FastAPI routes.
InferenceEngine = RealInferenceAdapter
