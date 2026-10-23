"""Sprint 0 data readiness demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from traffic_violation.data.labels import load_labels
from traffic_violation.data.yolo_dataset import summarize_yolo_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "00_data.txt"


def _coco128_lines() -> list[str]:
    dataset_root = ROOT / "data" / "coco128"
    if not dataset_root.is_dir():
        return ["COCO128: HAZIR DEĞİL"]
    config = yaml.safe_load((ROOT / "configs" / "coco128.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    class_names = [names[index] for index in range(len(names))]
    summary = summarize_yolo_dataset(
        dataset_root / "images" / "train2017",
        dataset_root / "labels" / "train2017",
        class_names,
    )
    relevant = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light")
    distribution = ", ".join(
        f"{name}={summary.class_counts.get(name, 0)}" for name in relevant
    )
    return [
        "COCO128: HAZIR (yalnızca boru hattı duman testi)",
        f"Görüntü={summary.image_count}; etiket={summary.label_file_count}; "
        f"nesne={summary.object_count}",
        f"İlgili sınıflar: {distribution}",
        "UYARI: train ve val aynı 128 görüntüdür; sonuç ürün/genelleme metriği değildir.",
    ]


def build_report(bdd_root: Path | None) -> str:
    example = load_labels(ROOT / "data" / "labels" / "example.json")
    lines = [
        "DEMO 00 — Veri ve lisans hazırlığı",
        f"Örnek etiket: {len(example)} olay; tür={example[0].kind}; t={example[0].frame_t:.1f}s",
    ]
    lines.extend(_coco128_lines())
    if bdd_root is None or not bdd_root.exists():
        lines.extend(
            [
                "BDD100K: HAZIR DEĞİL (beklenen ve güvenli durum)",
                "Resmî adres: https://www.bdd100k.com/",
                "İndirmeden önce lisansı data/LICENSES.md içinde doğrulayın.",
            ]
        )
    else:
        for split in ("train", "val"):
            folder = bdd_root / "images" / "100k" / split
            count = (
                sum(
                    1
                    for path in folder.glob("*")
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if folder.exists()
                else 0
            )
            lines.append(f"BDD100K {split}: {count} görüntü")
        lines.append("Sınıf dağılımı: etiket JSON'ları bağlandığında hesaplanacak.")
    license_text = (ROOT / "data" / "LICENSES.md").read_text(encoding="utf-8")
    lines.append(f"Lisans envanteri: {'UYARI VAR' if 'Belirsiz' in license_text else 'TAM'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-root", type=Path)
    args = parser.parse_args()
    output = build_report(args.bdd_root)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()

