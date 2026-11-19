from __future__ import annotations

import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, vocab: int, classes: int, architecture: str,
                 embedding: int = 64, hidden: int = 64) -> None:
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        if architecture == "cnn":
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding, hidden, kernel_size, padding=kernel_size // 2)
                for kernel_size in (3, 4, 5)
            ])
            output_size = hidden * 3
        elif architecture in ("bigru", "bilstm"):
            recurrent = nn.GRU if architecture == "bigru" else nn.LSTM
            self.rnn = recurrent(
                embedding, hidden, batch_first=True, bidirectional=True
            )
            output_size = hidden * 2
        elif architecture == "cnn_bigru_attention":
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding, hidden, kernel_size, padding=kernel_size // 2)
                for kernel_size in (3, 4, 5)
            ])
            self.rnn = nn.GRU(
                embedding, hidden, batch_first=True, bidirectional=True
            )
            self.attention = nn.Linear(hidden * 2, 1)
            output_size = hidden * 5
        else:
            raise ValueError(f"unknown architecture: {architecture}")
        self.head = nn.Linear(output_size, classes)

    def forward(self, ids):
        mask = ids.ne(0)
        embedded = self.embedding(ids)
        if self.architecture == "cnn":
            pooled = [
                nn.functional.relu(conv(embedded.transpose(1, 2))).amax(dim=2)
                for conv in self.convs
            ]
            features = torch.cat(pooled, dim=1)
        elif self.architecture in ("bigru", "bilstm"):
            output, _ = self.rnn(embedded)
            masked = output.masked_fill(~mask.unsqueeze(-1), -1e4)
            features = masked.amax(dim=1)
        else:
            cnn_features = torch.cat([
                nn.functional.relu(conv(embedded.transpose(1, 2))).amax(dim=2)
                for conv in self.convs
            ], dim=1)
            recurrent_output, _ = self.rnn(embedded)
            scores = self.attention(recurrent_output).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e4)
            weights = nn.functional.softmax(scores, dim=1)
            recurrent_features = torch.bmm(
                weights.unsqueeze(1), recurrent_output
            ).squeeze(1)
            features = torch.cat([cnn_features, recurrent_features], dim=1)
        return self.head(self.dropout(features))
