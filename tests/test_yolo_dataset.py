from pathlib import Path

import pytest

from traffic_violation.data.yolo_dataset import summarize_yolo_dataset


def test_summarize_yolo_dataset_counts_valid_rows(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").touch()
    (images / "b.png").touch()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.4 0.4 0.1 0.1\n")
    (labels / "b.txt").write_text("", encoding="utf-8")

    summary = summarize_yolo_dataset(images, labels, ["person", "car"])

    assert summary.image_count == 2
    assert summary.label_file_count == 2
    assert summary.annotated_image_count == 1
    assert summary.object_count == 2
    assert summary.class_counts == {"person": 1, "car": 1}


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 0.5 0.5 0.2", "expected 5 YOLO fields"),
        ("truck 0.5 0.5 0.2 0.2", "fields must be numeric"),
        ("2 0.5 0.5 0.2 0.2", "unknown class id"),
        ("0 1.1 0.5 0.2 0.2", "normalized coordinates"),
    ],
)
def test_summarize_yolo_dataset_rejects_invalid_rows(
    tmp_path: Path, row: str, message: str
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").touch()
    (labels / "a.txt").write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_yolo_dataset(images, labels, ["person", "car"])
