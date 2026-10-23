"""Demonstrate temporal event matching and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation.eval.plate_metrics import plate_accuracy
from traffic_violation.eval.violation_metrics import match_events, precision_recall_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "14_eval.txt"
CHART = ROOT / "reports" / "demos" / "14_eval.png"


def _write_chart(metrics: dict[str, int | float]) -> None:
    image = np.full((430, 760, 3), 34, dtype=np.uint8)
    cv2.putText(
        image, "Violation event evaluation", (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2,
    )
    for index, name in enumerate(("precision", "recall", "f1")):
        value = float(metrics[name])
        y = 130 + index * 95
        cv2.putText(
            image,
            name.upper(),
            (35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
        )
        cv2.rectangle(image, (190, y - 27), (690, y + 10), (65, 65, 65), -1)
        cv2.rectangle(image, (190, y - 27), (190 + int(500 * value), y + 10), (60, 180, 95), -1)
        cv2.putText(
            image, f"{value:.3f}", (615, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2,
        )
    if not cv2.imwrite(str(CHART), image):
        raise RuntimeError("could not write evaluation chart")


def main() -> None:
    gold = [
        {"kind": "red_light", "frame_t": 10.0},
        {"kind": "crosswalk", "frame_t": 20.0},
        {"kind": "red_light", "frame_t": 30.0},
    ]
    predictions = [
        {"kind": "red_light", "frame_t": 10.2},
        {"kind": "crosswalk", "frame_t": 20.4},
        {"kind": "stop_line", "frame_t": 30.0},
        {"kind": "red_light", "frame_t": 42.0},
    ]
    result = match_events(predictions, gold, tolerance_s=0.5)
    metrics = precision_recall_f1(result)
    optional_plate = plate_accuracy(["34 ABC 123"], ["34ABC123"])
    prediction_total = result.tp + result.fp
    actual_total = result.tp + result.fn
    false_alarm_rate = result.fp / prediction_total if prediction_total else 0.0
    miss_rate = result.fn / actual_total if actual_total else 0.0
    _write_chart(metrics)

    lines = [
        "DEMO 14 — Olay değerlendirme",
        "Veri: sentetik kabul senaryosu (gerçek klip baseline'ı değil)",
        "Tolerans: 0.5 s",
        f"TP: {result.tp}",
        f"FP: {result.fp}",
        f"FN: {result.fn}",
        f"Precision: {float(metrics['precision']):.3f}",
        f"Recall: {float(metrics['recall']):.3f}",
        f"F1: {float(metrics['f1']):.3f}",
        f"Tahmin doğruluğu: {result.tp}/{prediction_total} = "
        f"{float(metrics['precision']):.1%} (precision)",
        f"Yanlış alarm: {result.fp}/{prediction_total} = {false_alarm_rate:.1%}",
        f"Gerçek ihlal yakalama: {result.tp}/{actual_total} = "
        f"{float(metrics['recall']):.1%} (recall)",
        f"Kaçırılan ihlal: {result.fn}/{actual_total} = {miss_rate:.1%}",
        "Genel accuracy: hesaplanamaz; negatif klip/TN etiketi yok.",
        f"Eşleşmeyen tahmin indeksleri: {list(result.unmatched_pred)}",
        f"Eşleşmeyen etiket indeksleri: {list(result.unmatched_gold)}",
        f"Opsiyonel plaka normalizasyon testi: {optional_plate:.3f}",
        "Plaka tanıma ilk sürüm kapsamı dışıdır.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Grafik: {CHART}")
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()
