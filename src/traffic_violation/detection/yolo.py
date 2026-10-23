"""Lazy Ultralytics YOLO adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.types import Detection, filter_classes


class YoloDetector:
    """Load YOLO only on the first inference call and return domain detections."""

    def __init__(
        self,
        weights: str,
        classes: set[str],
        conf: float = 0.25,
        device: str = "cpu",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not weights:
            raise ValueError("weights cannot be empty")
        if not classes:
            raise ValueError("at least one allowed class is required")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.weights = weights
        self.classes = set(classes)
        self.conf = conf
        self.device = device
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.weights)
                return self._model
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed; install traffic-violation[ml]"
                ) from exc
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")
        results = self._ensure_model().predict(
            source=frame_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for coordinates, confidence, class_id in zip(
                xyxy, confidences, class_ids, strict=True
            ):
                index = int(class_id)
                class_name = names[index] if isinstance(names, dict) else names[index]
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                detections.append(
                    Detection(
                        cls=str(class_name),
                        conf=float(confidence),
                        box=(x1, y1, x2, y2),
                    )
                )
        return filter_classes(detections, self.classes)

