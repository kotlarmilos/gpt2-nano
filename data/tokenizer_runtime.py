from __future__ import annotations

import json
import os


TOKENIZER_PATH = "data/bpe-tokenizer"
EOW = "<|EOW|>"


def load_tokenizer() -> tuple[list[list[str]], dict[str, int]]:
    with open(
        os.path.join(TOKENIZER_PATH, "merges.json"), encoding="utf-8"
    ) as handle:
        merges = json.load(handle)
    with open(
        os.path.join(TOKENIZER_PATH, "vocab.json"), encoding="utf-8"
    ) as handle:
        vocab = json.load(handle)
    return merges, vocab


def encode(
    sentence: str, merges: list[list[str]], vocab: dict[str, int]
) -> list[int]:
    words = [list(word) + [EOW] for word in sentence.split(" ")]
    for first, second in merges:
        merged = first + second
        for word in words:
            index = 0
            while index < len(word) - 1:
                if word[index] == first and word[index + 1] == second:
                    word[index : index + 2] = [merged]
                else:
                    index += 1
    tokens = []
    for word in words:
        for token in word:
            if token not in vocab:
                raise ValueError(f"token {token!r} is not in the vocabulary")
            tokens.append(vocab[token])
    return tokens


def decode(tokens: list[int], vocab: dict[str, int]) -> str:
    inverse = {value: key for key, value in vocab.items()}
    pieces = []
    for token in tokens:
        if token not in inverse:
            raise ValueError(f"token id {token} is not in the vocabulary")
        piece = inverse[token]
        pieces.append(" " if piece == EOW else piece)
    return "".join(pieces)
