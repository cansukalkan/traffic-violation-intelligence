"""Download the small public smoke-test dataset with an integrity check."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def bootstrap(data_dir: Path = ROOT / "data") -> Path:
    dataset = data_dir / "coco128"
    if (dataset / "images" / "train2017").is_dir():
        print(f"COCO128 hazır: {dataset}")
        return dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        archive_path = Path(temporary) / "coco128.zip"
        print(f"COCO128 indiriliyor: {COCO128_URL}")
        with (
            urllib.request.urlopen(COCO128_URL, timeout=120) as response,
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        actual = sha256(archive_path)
        if actual != COCO128_SHA256:
            raise RuntimeError(f"COCO128 SHA-256 uyuşmuyor: {actual}")
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, data_dir)
    print(f"COCO128 hazır: {dataset}")
    return dataset


if __name__ == "__main__":
    bootstrap()
