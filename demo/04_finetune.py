"""Reproducible COCO128 baseline and one-epoch training smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_violation.eval.detection_metrics import DetectionMetrics, evaluate_detection
from traffic_violation.training.finetune_yolo import finetune

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "04_finetune.txt"


def _metric_lines(label: str, metrics: DetectionMetrics) -> list[str]:
    return [
        f"{label} mAP50: {metrics.map50:.4f}",
        f"{label} mAP50-95: {metrics.map50_95:.4f}",
        f"{label} precision: {metrics.precision:.4f}",
        f"{label} recall: {metrics.recall:.4f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    data_yaml = ROOT / "configs" / "coco128.yaml"
    baseline = evaluate_detection(args.weights, data_yaml, args.device, args.imgsz)
    lines = [
        "DEMO 04 — YOLO gerçek model duman testi",
        "Veri: COCO128 (128 görüntü; train ve val aynıdır)",
        f"Ağırlık: {args.weights}; cihaz: {args.device}; imgsz: {args.imgsz}",
        *_metric_lines("Baseline", baseline),
    ]

    if args.epochs > 0:
        best = finetune(
            data_yaml,
            args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=ROOT / "runs" / "detect",
            name="coco128-smoke",
            workers=0,
        )
        trained = evaluate_detection(best, data_yaml, args.device, args.imgsz)
        lines.extend([f"Smoke eğitim: {args.epochs} epoch", f"En iyi ağırlık: {best}"])
        lines.extend(_metric_lines("Eğitim sonrası", trained))
    else:
        lines.append("Smoke eğitim: çalıştırılmadı (--epochs 1 ile etkinleştirilir)")

    lines.extend(
        [
            "UYARI: Bu veri kümesinde train=val olduğundan metrikler genelleme veya ürün",
            "başarımı kanıtı değildir; yalnızca eğitim/değerlendirme hattının çalıştığını",
            "gösterir.",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()
