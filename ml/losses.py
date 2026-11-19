from __future__ import annotations

import torch
from torch import Tensor, nn


def emotion_stress_loss(emotion_logits: Tensor, emotion_targets: Tensor,
                        stress_logits: Tensor, stress_targets: Tensor,
                        emotion_weight: float = 1.0, stress_weight: float = 1.0) -> Tensor:
    emotion_loss = nn.functional.cross_entropy(emotion_logits, emotion_targets)
    stress_loss = nn.functional.binary_cross_entropy_with_logits(stress_logits, stress_targets.float())
    return emotion_weight * emotion_loss + stress_weight * stress_loss
