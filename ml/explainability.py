from __future__ import annotations

from typing import Callable, Dict, List


def occlusion_explanation(text: str, analyzer: Callable[[str], Dict],
                          max_tokens: int = 12) -> List[Dict[str, object]]:
    """Token occlusion: change is removed-score minus occluded-score."""
    words = text.split()
    baseline = analyzer(text)
    base_emotion = float(max(baseline["emotions"], key=lambda item: item["score"])["score"])
    base_stress = float(baseline["stress_score"])
    base_value = base_emotion + base_stress
    results: List[Dict[str, object]] = []
    for index, word in enumerate(words[:max_tokens]):
        occluded = " ".join(words[:index] + words[index + 1:])
        if not occluded:
            continue
        candidate = analyzer(occluded)
        cand_emotion = float(max(candidate["emotions"], key=lambda item: item["score"])["score"])
        importance = base_value - (cand_emotion + float(candidate["stress_score"]))
        results.append({"text": word, "importance": round(importance, 6)})
    return sorted(results, key=lambda item: abs(float(item["importance"])), reverse=True)


def uncertainty_from_scores(probabilities: List[float], entropy_weight: float = 0.65) -> Dict[str, object]:
    import math

    if not probabilities:
        return {"confidence": 0.0, "is_uncertain": True, "method": "normalized_entropy"}
    normalized = [max(0.0, min(1.0, float(value))) for value in probabilities]
    total = sum(normalized)
    normalized = [value / total for value in normalized] if total else [1.0 / len(normalized)] * len(normalized)
    entropy = -sum(value * math.log(value + 1e-12) for value in normalized)
    max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - entropy_weight * (entropy / max_entropy)))
    return {"confidence": round(confidence, 6), "is_uncertain": confidence < 0.65, "method": "normalized_entropy"}
