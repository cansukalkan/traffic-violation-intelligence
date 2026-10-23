import json
from pathlib import Path

import pytest

from traffic_violation.data.bdd_subset import make_bdd_subset
from traffic_violation.data.labels import load_labels, validate_label


def test_validate_label_accepts_good_record() -> None:
    label = validate_label({"kind": "red_light", "frame_t": 12.3, "track_id": 4})
    assert label.kind == "red_light"


def test_validate_label_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        validate_label({"kind": "speeding", "frame_t": 1.0, "track_id": 1})


def test_load_labels_validates_envelope(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "clip": "clip.mp4",
                "fps": 10,
                "events": [{"kind": "crosswalk", "frame_t": 1.2, "track_id": 8}],
            }
        ),
        encoding="utf-8",
    )
    assert load_labels(path)[0].track_id == 8


def test_subset_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "bdd"
    for split, count in (("train", 9), ("val", 4)):
        directory = source / "images" / "100k" / split
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index:03}.jpg").touch()
    first = make_bdd_subset(source, tmp_path / "a", n_train=5, n_val=2, seed=42)
    second = make_bdd_subset(source, tmp_path / "b", n_train=5, n_val=2, seed=42)
    assert first.train_files == second.train_files
    assert first.val_files == second.val_files
    assert len((tmp_path / "a" / "train.txt").read_text(encoding="utf-8").splitlines()) == 5


def test_subset_rejects_oversampling(tmp_path: Path) -> None:
    for split in ("train", "val"):
        directory = tmp_path / "bdd" / split
        directory.mkdir(parents=True)
        (directory / "one.jpg").touch()
    with pytest.raises(ValueError, match="only 1"):
        make_bdd_subset(tmp_path / "bdd", tmp_path / "out", 2, 1, 42)

