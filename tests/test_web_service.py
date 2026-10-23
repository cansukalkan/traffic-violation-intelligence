from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_violation.web.service import (
    MediaAnalysisService,
    MediaValidationError,
    SampleCatalog,
)


def _jpeg_bytes() -> bytes:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:, :60] = (20, 90, 180)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(encoded)


def test_analyze_image_persists_job_and_preview(tmp_path: Path) -> None:
    service = MediaAnalysisService(tmp_path)

    job = service.analyze_upload(io.BytesIO(_jpeg_bytes()), "../camera 01.jpg")

    assert job.original_name == "camera 01.jpg"
    assert job.media_kind == "image"
    assert (job.width, job.height) == (120, 80)
    assert job.mode == "source"
    assert job.detection_count == 0
    assert (service.preview_dir / job.preview_name).is_file()
    assert service.store.get(job.id) == job
    assert service.store.list() == [job]


@pytest.mark.parametrize("name", ["payload.exe", "archive.zip", "no-extension"])
def test_rejects_unsupported_extension(tmp_path: Path, name: str) -> None:
    service = MediaAnalysisService(tmp_path)

    with pytest.raises(MediaValidationError, match="Yalnızca"):
        service.analyze_upload(io.BytesIO(b"unsafe"), name)


def test_rejects_invalid_image_and_removes_upload(tmp_path: Path) -> None:
    service = MediaAnalysisService(tmp_path)

    with pytest.raises(MediaValidationError, match="çözümlenemedi"):
        service.analyze_upload(io.BytesIO(b"not-an-image"), "broken.jpg")

    assert list(service.upload_dir.iterdir()) == []


def test_store_limit_is_bounded(tmp_path: Path) -> None:
    service = MediaAnalysisService(tmp_path)

    with pytest.raises(ValueError, match="between 1 and 100"):
        service.store.list(101)


def test_sample_catalog_is_a_read_only_allowlist(tmp_path: Path) -> None:
    (tmp_path / "one.jpg").write_bytes(_jpeg_bytes())
    (tmp_path / "ignored.txt").write_text("not media", encoding="utf-8")
    catalog = SampleCatalog(tmp_path)

    assert [item.id for item in catalog.list()] == ["one"]
    assert catalog.get("one") is not None
    assert catalog.get("../one") is None


def test_onnx_mode_returns_traffic_detections(tmp_path: Path) -> None:
    class FakeNet:
        def setInput(self, _blob: np.ndarray) -> None:  # noqa: N802
            pass

        def forward(self) -> np.ndarray:
            output = np.zeros((1, 84, 2100), dtype=np.float32)
            output[0, :4, 0] = [160, 160, 100, 80]
            output[0, 6, 0] = 0.92  # COCO class 2: car
            return output

    service = MediaAnalysisService(tmp_path, mode="onnx")
    service._onnx_net = FakeNet()

    job = service.analyze_upload(io.BytesIO(_jpeg_bytes()), "traffic.jpg")

    assert job.detection_count == 1
    assert job.detected_classes == ["car"]
