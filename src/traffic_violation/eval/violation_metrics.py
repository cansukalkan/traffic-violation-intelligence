"""One-to-one temporal matching and violation classification metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchResult:
    tp: int
    fp: int
    fn: int
    matches: tuple[tuple[int, int], ...]
    unmatched_pred: tuple[int, ...]
    unmatched_gold: tuple[int, ...]


def _record(record: Mapping[str, object]) -> tuple[str, float]:
    kind = record.get("kind")
    frame_t = record.get("frame_t")
    if not isinstance(kind, str) or not kind:
        raise ValueError("each event requires a non-empty kind")
    if (
        not isinstance(frame_t, (int, float))
        or isinstance(frame_t, bool)
        or float(frame_t) < 0
    ):
        raise ValueError("each event requires a non-negative frame_t")
    return kind, float(frame_t)


def match_events(
    pred: Sequence[Mapping[str, object]],
    gold: Sequence[Mapping[str, object]],
    tolerance_s: float,
) -> MatchResult:
    """Maximize one-to-one matches for events of the same kind within tolerance."""

    if tolerance_s < 0:
        raise ValueError("tolerance_s cannot be negative")
    pred_records = [(*_record(item), index) for index, item in enumerate(pred)]
    gold_records = [(*_record(item), index) for index, item in enumerate(gold)]
    matches: list[tuple[int, int]] = []

    for kind in sorted({item[0] for item in pred_records + gold_records}):
        pred_kind = sorted((item for item in pred_records if item[0] == kind), key=lambda x: x[1])
        gold_kind = sorted((item for item in gold_records if item[0] == kind), key=lambda x: x[1])
        pred_index = gold_index = 0
        while pred_index < len(pred_kind) and gold_index < len(gold_kind):
            _, pred_time, original_pred = pred_kind[pred_index]
            _, gold_time, original_gold = gold_kind[gold_index]
            if pred_time < gold_time - tolerance_s:
                pred_index += 1
            elif gold_time < pred_time - tolerance_s:
                gold_index += 1
            else:
                matches.append((original_pred, original_gold))
                pred_index += 1
                gold_index += 1

    matched_pred = {item[0] for item in matches}
    matched_gold = {item[1] for item in matches}
    unmatched_pred = tuple(index for index in range(len(pred)) if index not in matched_pred)
    unmatched_gold = tuple(index for index in range(len(gold)) if index not in matched_gold)
    return MatchResult(
        tp=len(matches),
        fp=len(unmatched_pred),
        fn=len(unmatched_gold),
        matches=tuple(sorted(matches)),
        unmatched_pred=unmatched_pred,
        unmatched_gold=unmatched_gold,
    )


def precision_recall_f1(result: MatchResult) -> dict[str, int | float]:
    precision = result.tp / (result.tp + result.fp) if result.tp + result.fp else 0.0
    recall = result.tp / (result.tp + result.fn) if result.tp + result.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
