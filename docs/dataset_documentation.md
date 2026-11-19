# Dataset documentation

The intended research sources are GoEmotions for 28-label emotion classification, Dreaddit for binary stress-related language classification, and EmpatheticDialogues for ordered contextual emotion experiments. Their labels are not interchangeable; the training pipeline keeps task-specific batches and evaluates tasks separately. EmpatheticDialogues is never treated as stress ground truth.

The acquisition script supports documented local/Hugging Face workflows. Raw datasets are not committed. Before training, validate missing text, duplicates, labels, class imbalance, conversation IDs, empty conversations, and split assignments. Vocabulary construction must use training data only, and the maximum message length is 160 tokens. The synthetic fixture exists only to test software plumbing and must not be used as scientific evaluation data.
