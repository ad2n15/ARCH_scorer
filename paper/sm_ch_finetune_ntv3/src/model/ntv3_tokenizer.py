"""Offline character tokenizer for NTv3 (uses vocab.json from model snapshot)."""
from __future__ import annotations

import json
from pathlib import Path

from transformers import PreTrainedTokenizer


class NTv3CharTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str | Path, **kwargs):
        with open(vocab_file) as f:
            vocab = json.load(f)
        self._vocab = vocab
        self._ids_to_tokens = {i: t for t, i in vocab.items()}
        super().__init__(
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            cls_token="<cls>",
            eos_token="<eos>",
            bos_token="<bos>",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        path = Path(save_directory) / f"{filename_prefix or ''}vocab.json"
        with open(path, "w") as f:
            json.dump(self._vocab, f, indent=2)
        return (str(path),)


def load_char_tokenizer(model_path: str | Path) -> NTv3CharTokenizer:
    model_path = Path(model_path)
    vocab = model_path / "vocab.json"
    if not vocab.is_file():
        raise FileNotFoundError(f"Missing {vocab}")
    return NTv3CharTokenizer(vocab)
