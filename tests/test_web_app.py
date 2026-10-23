from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from traffic_violation.web.app import create_app


def _png_bytes() -> bytes:
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return bytes(encoded)


def test_dashboard_upload_flow(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path, mode="source")) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Traffic Violation Intelligence" in home.text

        response = client.post(
            "/api/analyze",
            files={"file": ("junction.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 201
        job = response.json()
        assert job["original_name"] == "junction.png"
        assert job["width"] == 64
        assert job["height"] == 48

        listing = client.get("/api/jobs").json()
        assert listing["count"] == 1
        assert listing["items"][0]["id"] == job["id"]

        preview = client.get(f"/api/jobs/{job['id']}/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"


def test_health_reports_runtime_mode(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path, mode="source")) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "storage": "sqlite",
            "analyzer_mode": "source",
        }


def test_upload_validation_error_is_422(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path, mode="source")) as client:
        response = client.post(
            "/api/analyze",
            files={"file": ("malware.exe", b"payload", "application/octet-stream")},
        )
    assert response.status_code == 422


def test_missing_job_is_404(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path, mode="source")) as client:
        assert client.get("/api/jobs/missing").status_code == 404


def test_bundled_sample_can_be_selected_and_analyzed(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "demo.png").write_bytes(_png_bytes())
    data_dir = tmp_path / "runtime"

    with TestClient(
        create_app(data_dir=data_dir, mode="source", sample_dir=sample_dir)
    ) as client:
        catalog = client.get("/api/samples").json()
        assert catalog["count"] == 1
        assert catalog["items"][0]["id"] == "demo"
        assert client.get("/api/samples/demo/thumbnail").status_code == 200

        response = client.post("/api/samples/demo/analyze")
        assert response.status_code == 201
        assert response.json()["original_name"] == "sample-demo.png"
        assert client.get("/api/samples/..%2Fdemo/thumbnail").status_code == 404
