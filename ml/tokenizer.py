from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence


class WordTokenizer:
    """Compact whitespace/punctuation tokenizer; vocabulary is local and dependency-free."""

    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self, vocab_size: int = 30000) -> None:
        self.vocab_size = vocab_size
        self.itos = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @staticmethod
    def normalize(text: str) -> List[str]:
        return re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*|\d+", text.lower(), flags=re.UNICODE)

    def fit(self, texts: Iterable[str]) -> "WordTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.normalize(text):
                counts[token] = counts.get(token, 0) + 1
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if len(self.itos) >= self.vocab_size:
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self

    def encode(self, text: str, max_length: int = 128, add_special: bool = True) -> List[int]:
        tokens = self.normalize(text)
        if add_special:
            tokens = [self.BOS] + tokens + [self.EOS]
        ids = [self.stoi.get(token, self.stoi[self.UNK]) for token in tokens[:max_length]]
        if len(ids) < max_length:
            ids.extend([self.stoi[self.PAD]] * (max_length - len(ids)))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        special = {self.stoi[self.PAD], self.stoi[self.BOS], self.stoi[self.EOS]}
        return " ".join(
            self.itos[index] if 0 <= index < len(self.itos) else self.UNK
            for index in ids
            if index not in special
        )

    def save(self, path: str | Path) -> None:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"vocab_size": self.vocab_size, "itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(data["vocab_size"])
        tokenizer.itos = data["itos"]
        tokenizer.stoi = {token: index for index, token in enumerate(tokenizer.itos)}
        return tokenizer
