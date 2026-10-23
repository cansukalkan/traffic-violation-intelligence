"""Deterministic BDD100K subset manifests without copying source images."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubsetInfo:
    train_files: list[str]
    val_files: list[str]


def _discover_split(root: Path, split: str) -> list[Path]:
    candidates = (root / "images" / "100k" / split, root / split)
    split_dir = next((path for path in candidates if path.is_dir()), None)
    if split_dir is None:
        expected = " or ".join(str(path) for path in candidates)
        raise ValueError(f"BDD100K {split!r} image directory not found; expected {expected}")
    files = sorted(
        path.resolve()
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise ValueError(f"no images found in {split_dir}")
    return files


def _sample(files: list[Path], count: int, seed: int, split: str) -> list[Path]:
    if count < 0:
        raise ValueError(f"{split} sample count cannot be negative")
    if count > len(files):
        raise ValueError(f"requested {count} {split} images but only {len(files)} are available")
    rng = random.Random(f"{seed}:{split}")
    return sorted(rng.sample(files, count))


def make_bdd_subset(
    src_dir: str | Path,
    out_dir: str | Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> SubsetInfo:
    """Write deterministic train/val manifests; never copy image data."""

    source = Path(src_dir)
    output = Path(out_dir)
    train = _sample(_discover_split(source, "train"), n_train, seed, "train")
    val = _sample(_discover_split(source, "val"), n_val, seed, "val")
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in train), encoding="utf-8"
    )
    (output / "val.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in val), encoding="utf-8"
    )
    return SubsetInfo(
        train_files=[path.as_posix() for path in train],
        val_files=[path.as_posix() for path in val],
    )

