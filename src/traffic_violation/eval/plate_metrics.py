"""Optional plate-string normalization metric; plate recognition remains out of scope."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def normalize_plate(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).upper()
    return "".join(character for character in normalized if character.isalnum())


def plate_accuracy(pred: Sequence[str | None], gold: Sequence[str | None]) -> float:
    total = max(len(pred), len(gold))
    if total == 0:
        return 1.0
    correct = 0
    for index in range(total):
        predicted = pred[index] if index < len(pred) else None
        expected = gold[index] if index < len(gold) else None
        if predicted is not None and expected is not None:
            correct += normalize_plate(predicted) == normalize_plate(expected)
    return correct / total
