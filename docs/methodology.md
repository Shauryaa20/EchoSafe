# Methodology

The proposed encoder uses token embeddings, parallel kernel-3/kernel-5 CNN branches, and a bidirectional GRU with masked attention pooling. Message representations are passed through a masked context LSTM over up to six ordered messages, followed by attention pooling and a 128-unit shared representation. Emotion uses a 28-label sigmoid head; stress language uses a separate sigmoid head. Because source datasets have different schemas and samples, training uses task-specific batches and masked losses rather than fabricated alignment.

Explanations use token occlusion: the service measures the change in the selected score when a token is removed. Uncertainty combines prediction entropy and confidence thresholds calibrated from validation artifacts when available. It is not a clinical threshold.
