from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, values: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        logits = self.score(torch.tanh(self.query(values))).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights)) if mask is not None else weights
        pooled = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return pooled, weights


class MessageEncoder(nn.Module):
    """Parallel 1-D CNN and bidirectional GRU encoder with attention pooling."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128,
                 cnn_channels: int = 128, cnn_kernel_sizes: Tuple[int, ...] = (3, 4, 5),
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, cnn_channels, kernel_size, padding=kernel_size // 2)
            for kernel_size in cnn_kernel_sizes
        ])
        self.bigru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        bigru_dim = hidden_size * 2
        self.attention = Attention(bigru_dim)
        self.projection = nn.Linear(cnn_channels * len(cnn_kernel_sizes) + bigru_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        embedded = self.embedding(token_ids)
        cnn_input = embedded.transpose(1, 2)
        cnn_features = [torch.relu(conv(cnn_input)).mean(dim=2) for conv in self.convs]
        cnn_pooled = torch.cat(cnn_features, dim=1)
        lengths = mask.long().sum(dim=1).clamp(min=1).cpu() if mask is not None else None
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        ) if lengths is not None else embedded
        gru_output, _ = self.bigru(packed)
        if lengths is not None:
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True, total_length=token_ids.size(1))
        gru_output = torch.zeros_like(gru_output) if gru_output is None else gru_output
        gru_pooled, _ = self.attention(gru_output, mask)
        combined = torch.cat([cnn_pooled, gru_pooled], dim=1)
        return self.dropout(torch.relu(self.projection(combined)))


class EchoSafeModel(nn.Module):
    """CNN + BiGRU message encoder followed by masked context LSTM and task heads."""

    def __init__(self, vocab_size: int, emotion_classes: int = 7, embedding_dim: int = 128,
                 message_hidden: int = 128, context_hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.emotion_classes = emotion_classes
        self.message_encoder = MessageEncoder(vocab_size, embedding_dim, message_hidden, dropout=dropout)
        self.context_lstm = nn.LSTM(message_hidden, context_hidden, batch_first=True, bidirectional=False)
        self.context_attention = Attention(context_hidden)
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(context_hidden, emotion_classes)
        self.stress_head = nn.Linear(context_hidden, 1)

    def forward(self, token_ids: Tensor, lengths: Optional[Tensor] = None,
                return_attention: bool = False) -> Dict[str, Tensor]:
        batch, conversation, sequence = token_ids.shape
        flat_ids = token_ids.reshape(batch * conversation, sequence)
        flat_mask = flat_ids.ne(0)
        message_vectors = self.message_encoder(flat_ids, flat_mask)
        message_vectors = message_vectors.reshape(batch, conversation, -1)

        if lengths is not None:
            if lengths.dim() == 1:
                positions = torch.arange(conversation, device=token_ids.device).unsqueeze(0)
                valid_mask = positions < lengths.to(token_ids.device).unsqueeze(1)
            else:
                valid_mask = lengths.ne(0).to(token_ids.device)
        else:
            valid_mask = torch.ones(batch, conversation, dtype=torch.bool, device=token_ids.device)
        max_steps = valid_mask.sum(dim=1).max().long().item() if valid_mask.numel() else 0
        if max_steps == 0:
            raise ValueError("at least one message is required")
        packed_context = message_vectors[:, :max_steps]
        context_output, _ = self.context_lstm(packed_context)
        context_pooled, context_weights = self.context_attention(context_output, valid_mask[:, :max_steps])
        features = self.dropout(context_pooled)
        emotion_logits = self.emotion_head(features)
        stress_logits = self.stress_head(features).squeeze(-1)
        result = {
            "emotion_logits": emotion_logits,
            "emotion_probs": torch.softmax(emotion_logits, dim=-1),
            "stress_logits": stress_logits,
            "stress_prob": torch.sigmoid(stress_logits),
        }
        if return_attention:
            result["context_attention"] = context_weights
            result["message_embeddings"] = message_vectors
        return result
