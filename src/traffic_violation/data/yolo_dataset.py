"""Small, dependency-free checks for YOLO detection datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloDatasetSummary:
    """Counts produced after validating all YOLO label rows."""

    image_count: int
    label_file_count: int
    annotated_image_count: int
    object_count: int
    class_counts: dict[str, int]


def summarize_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: Sequence[str],
) -> YoloDatasetSummary:
    """Validate YOLO labels and summarize images, annotations and classes."""

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not label_root.is_dir():
        raise FileNotFoundError(label_root)
    if not class_names:
        raise ValueError("class_names cannot be empty")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [path for path in image_root.rglob("*") if path.suffix.lower() in image_suffixes]
    labels = sorted(label_root.rglob("*.txt"))
    counts: Counter[int] = Counter()
    annotated_images = 0

    for label_path in labels:
        has_annotation = False
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
            try:
                class_id = int(fields[0])
                coordinates = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"{label_path}:{line_number}: fields must be numeric"
                ) from exc
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"{label_path}:{line_number}: unknown class id {class_id}")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(
                    f"{label_path}:{line_number}: normalized coordinates must be in [0, 1]"
                )
            counts[class_id] += 1
            has_annotation = True
        annotated_images += int(has_annotation)

    named_counts = {
        class_names[class_id]: count
        for class_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }
    return YoloDatasetSummary(
        image_count=len(images),
        label_file_count=len(labels),
        annotated_image_count=annotated_images,
        object_count=sum(counts.values()),
        class_counts=named_counts,
    )
