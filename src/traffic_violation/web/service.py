"""Safe media ingestion and honest first-pass analysis for the dashboard."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
TRAFFIC_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}
PREFERRED_SAMPLE_IDS = (
    "000000000257",
    "000000000064",
    "000000000073",
    "000000000036",
    "000000000049",
    "000000000061",
    "000000000074",
    "000000000077",
    "000000000086",
    "000000000094",
    "000000000109",
    "000000000110",
)


class MediaValidationError(ValueError):
    """Raised when an upload is unsafe, unsupported, or undecodable."""


@dataclass(frozen=True, slots=True)
class SampleMedia:
    id: str
    name: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "thumbnail_url": f"/api/samples/{self.id}/thumbnail",
        }


class SampleCatalog:
    """Read-only allowlist of bundled demonstration media."""

    def __init__(self, root: Path, *, limit: int = 12) -> None:
        self.root = root.resolve()
        discovered = [
            path
            for path in sorted(self.root.glob("*"))
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ]
        preferred_order = {
            sample_id: index for index, sample_id in enumerate(PREFERRED_SAMPLE_IDS)
        }
        discovered.sort(
            key=lambda path: (
                preferred_order.get(path.stem, len(preferred_order)),
                path.name,
            )
        )
        self._items = {
            path.stem: SampleMedia(id=path.stem, name=path.name, path=path)
            for path in discovered[:limit]
        }

    def list(self) -> list[SampleMedia]:
        return list(self._items.values())

    def get(self, sample_id: str) -> SampleMedia | None:
        return self._items.get(sample_id)


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    id: str
    original_name: str
    media_kind: str
    status: str
    mode: str
    width: int
    height: int
    frame_count: int
    fps: float
    duration_s: float
    motion_score: float
    detection_count: int
    detected_classes: list[str]
    preview_name: str
    created_at: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    media_kind: str
    width: int
    height: int
    frame_count: int
    fps: float
    duration_s: float
    motion_score: float


class AnalysisStore:
    """Small WAL-mode SQLite store dedicated to uploaded analysis jobs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    media_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    motion_score REAL NOT NULL,
                    detection_count INTEGER NOT NULL,
                    detected_classes_json TEXT NOT NULL,
                    preview_name TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def append(self, job: AnalysisJob) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO analysis_jobs VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.original_name,
                    job.media_kind,
                    job.status,
                    job.mode,
                    job.width,
                    job.height,
                    job.frame_count,
                    job.fps,
                    job.duration_s,
                    job.motion_score,
                    job.detection_count,
                    json.dumps(job.detected_classes, ensure_ascii=False),
                    job.preview_name,
                    job.created_at,
                ),
            )

    def list(self, limit: int = 25) -> list[AnalysisJob]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM analysis_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def get(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return None if row is None else self._row(row)

    @staticmethod
    def _row(row: sqlite3.Row) -> AnalysisJob:
        return AnalysisJob(
            id=str(row["id"]),
            original_name=str(row["original_name"]),
            media_kind=str(row["media_kind"]),
            status=str(row["status"]),
            mode=str(row["mode"]),
            width=int(row["width"]),
            height=int(row["height"]),
            frame_count=int(row["frame_count"]),
            fps=float(row["fps"]),
            duration_s=float(row["duration_s"]),
            motion_score=float(row["motion_score"]),
            detection_count=int(row["detection_count"]),
            detected_classes=list(json.loads(row["detected_classes_json"])),
            preview_name=str(row["preview_name"]),
            created_at=float(row["created_at"]),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class MediaAnalysisService:
    """Persists uploads, validates decoding, and creates an annotated preview."""

    def __init__(self, root: Path, *, mode: str = "source") -> None:
        self.root = root.resolve()
        self.upload_dir = self.root / "uploads"
        self.preview_dir = self.root / "previews"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._onnx_net: Any | None = None
        self.store = AnalysisStore(self.root / "dashboard.db")

    def analyze_upload(self, stream: BinaryIO, original_name: str) -> AnalysisJob:
        suffix = Path(original_name).suffix.casefold()
        if suffix not in ALLOWED_EXTENSIONS:
            raise MediaValidationError("Yalnızca JPG, PNG, MP4, AVI, MOV veya MKV kabul edilir.")
        job_id = uuid.uuid4().hex
        upload_path = self.upload_dir / f"{job_id}{suffix}"
        try:
            self._copy_limited(stream, upload_path)
            frame, metadata = self._decode(upload_path, suffix)
            detections = self._detect(frame)
            preview_name = f"{job_id}.jpg"
            self._write_preview(frame, self.preview_dir / preview_name, detections, metadata)
            job = AnalysisJob(
                id=job_id,
                original_name=Path(original_name).name[:180],
                media_kind=metadata.media_kind,
                status="completed",
                mode=self.mode,
                width=metadata.width,
                height=metadata.height,
                frame_count=metadata.frame_count,
                fps=metadata.fps,
                duration_s=metadata.duration_s,
                motion_score=metadata.motion_score,
                detection_count=len(detections),
                detected_classes=sorted({item[0] for item in detections}),
                preview_name=preview_name,
                created_at=time.time(),
            )
            self.store.append(job)
            return job
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _copy_limited(stream: BinaryIO, destination: Path) -> None:
        total = 0
        with destination.open("wb") as output:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise MediaValidationError("Dosya 100 MB sınırını aşıyor.")
                output.write(chunk)
        if total == 0:
            raise MediaValidationError("Boş dosya yüklenemez.")

    def _decode(self, path: Path, suffix: str) -> tuple[np.ndarray, MediaMetadata]:
        if suffix in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(path))
            if frame is None:
                raise MediaValidationError("Görsel çözümlenemedi veya dosya bozuk.")
            height, width = frame.shape[:2]
            return frame, MediaMetadata(
                media_kind="image",
                width=width,
                height=height,
                frame_count=1,
                fps=0.0,
                duration_s=0.0,
                motion_score=0.0,
            )

        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise MediaValidationError(
                    "Video çözümlenemedi veya desteklenmeyen codec içeriyor."
                )
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
            duration = frame_count / fps if fps > 0 else 0.0
            motion = self._motion_score(capture, frame, frame_count)
            return frame, MediaMetadata(
                media_kind="video",
                width=width,
                height=height,
                frame_count=frame_count,
                fps=fps,
                duration_s=duration,
                motion_score=motion,
            )
        finally:
            capture.release()

    @staticmethod
    def _motion_score(capture: cv2.VideoCapture, first: np.ndarray, frame_count: int) -> float:
        previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        scores: list[float] = []
        for index in np.linspace(1, max(1, frame_count - 1), num=min(8, frame_count), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, current = capture.read()
            if not ok or current is None:
                continue
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            if gray.shape != previous.shape:
                continue
            scores.append(float(np.mean(cv2.absdiff(previous, gray))) / 255.0)
            previous = gray
        return round(float(np.mean(scores)) if scores else 0.0, 4)

    def _detect(self, frame: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        if self.mode == "source":
            return []
        model_path = os.environ.get("TV_MODEL_PATH", "yolo11n.pt")
        if self.mode == "onnx":
            return self._detect_onnx(frame, model_path)
        try:
            from traffic_violation.detection.yolo import YoloDetector

            found = YoloDetector(
                weights=model_path,
                classes={"car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light"},
            ).detect(frame)
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise RuntimeError("YOLO modu seçili ancak ML bağımlılıkları kurulu değil.") from exc
        result: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for item in found:
            x1, y1, x2, y2 = (int(value) for value in item.box)
            result.append((item.cls, item.conf, (x1, y1, x2, y2)))
        return result

    def _detect_onnx(
        self, frame: np.ndarray, model_path: str
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        if self._onnx_net is None:
            if not Path(model_path).is_file():
                raise RuntimeError(f"ONNX model dosyası bulunamadı: {model_path}")
            self._onnx_net = cv2.dnn.readNetFromONNX(model_path)
        input_size = 320
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255.0, (input_size, input_size), swapRB=True, crop=False
        )
        self._onnx_net.setInput(blob)
        output = np.asarray(self._onnx_net.forward())
        if output.ndim != 3 or output.shape[1] != 84:
            raise RuntimeError(f"Beklenmeyen YOLO ONNX çıktısı: {output.shape}")
        predictions = output[0].T
        height, width = frame.shape[:2]
        x_scale, y_scale = width / input_size, height / input_size
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in predictions:
            class_id = int(np.argmax(row[4:]))
            score = float(row[4 + class_id])
            if score < 0.25 or COCO_CLASSES[class_id] not in TRAFFIC_CLASSES:
                continue
            cx, cy, box_width, box_height = (float(value) for value in row[:4])
            left = int((cx - box_width / 2) * x_scale)
            top = int((cy - box_height / 2) * y_scale)
            boxes.append(
                [left, top, int(box_width * x_scale), int(box_height * y_scale)]
            )
            scores.append(score)
            class_ids.append(class_id)
        kept = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        result: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for index in np.asarray(kept).reshape(-1):
            left, top, box_width, box_height = boxes[int(index)]
            x1, y1 = max(0, left), max(0, top)
            x2, y2 = min(width - 1, left + box_width), min(height - 1, top + box_height)
            if x2 > x1 and y2 > y1:
                result.append(
                    (
                        COCO_CLASSES[class_ids[int(index)]],
                        scores[int(index)],
                        (x1, y1, x2, y2),
                    )
                )
        return result

    @staticmethod
    def _write_preview(
        frame: np.ndarray,
        destination: Path,
        detections: list[tuple[str, float, tuple[int, int, int, int]]],
        metadata: MediaMetadata,
    ) -> None:
        canvas = frame.copy()
        for label, confidence, (x1, y1, x2, y2) in detections:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 220, 150), 2)
            cv2.putText(
                canvas,
                f"{label} {confidence:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (50, 220, 150),
                2,
            )
        label = f"{metadata.media_kind} | {metadata.width}x{metadata.height}"
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 520), 42), (12, 18, 29), -1)
        cv2.putText(canvas, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 244, 255), 2)
        if not cv2.imwrite(str(destination), canvas):
            raise RuntimeError("Önizleme yazılamadı.")
