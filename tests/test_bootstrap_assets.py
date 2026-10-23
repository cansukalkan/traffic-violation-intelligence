from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_assets.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap_assets)


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"traffic-violation")

    assert bootstrap_assets.sha256(source) == (
        "b488698ac409411fcce21efb120434d39a88d6b9b244e1ea9fc40648b1385385"
    )


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="unsafe archive member"),
    ):
        bootstrap_assets.safe_extract(archive, tmp_path / "output")
