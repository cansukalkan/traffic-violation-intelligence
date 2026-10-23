from pathlib import Path

import numpy as np
import pytest

from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.types import Detection, filter_classes
from traffic_violation.detection.yolo import YoloDetector
from traffic_violation.eval.detection_metrics import evaluate_detection
from traffic_violation.training.finetune_yolo import finetune


def test_fake_detector_scripted() -> None:
    detector = FakeDetector([[Detection("car", 0.9, (0, 0, 10, 10))], []])
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert len(detector.detect(frame)) == 1
    assert detector.detect(frame) == []


def test_fake_detector_returns_empty_after_script() -> None:
    detector = FakeDetector([])
    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []


def test_filter_classes() -> None:
    detections = [
        Detection("car", 0.9, (0, 0, 1, 1)),
        Detection("cat", 0.9, (0, 0, 1, 1)),
    ]
    assert [item.cls for item in filter_classes(detections, {"car"})] == ["car"]


def test_detection_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection("car", 1.5, (0, 0, 10, 10))
    with pytest.raises(ValueError, match="positive"):
        Detection("car", 0.5, (10, 0, 1, 10))


def test_yolo_configuration_validates_without_loading_model() -> None:
    detector = YoloDetector("not-downloaded.pt", {"car"}, conf=0.5, device="cpu")
    assert detector.loaded is False
    with pytest.raises(ValueError, match="confidence"):
        YoloDetector("x.pt", {"car"}, conf=2.0)


def test_yolo_maps_model_output_without_real_model() -> None:
    class TensorStub:
        def __init__(self, values: object) -> None:
            self.values = values

        def detach(self) -> "TensorStub":
            return self

        def cpu(self) -> "TensorStub":
            return self

        def tolist(self) -> object:
            return self.values

    class BoxesStub:
        xyxy = TensorStub([[1.0, 2.0, 11.0, 22.0], [0.0, 0.0, 5.0, 5.0]])
        conf = TensorStub([0.91, 0.88])
        cls = TensorStub([0.0, 1.0])

    class ResultStub:
        boxes = BoxesStub()
        names = {0: "car", 1: "cat"}

    class ModelStub:
        def predict(self, **kwargs: object) -> list[ResultStub]:
            assert kwargs["device"] == "cpu"
            return [ResultStub()]

    detector = YoloDetector(
        "stub.pt", {"car"}, conf=0.4, model_factory=lambda _: ModelStub()
    )
    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert detections == [Detection("car", 0.91, (1.0, 2.0, 11.0, 22.0))]
    assert detector.loaded is True


def test_finetune_validates_inputs_before_loading_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs"):
        finetune(tmp_path / "data.yaml", "base.pt", epochs=0, imgsz=640, device="cpu")
    with pytest.raises(FileNotFoundError):
        finetune(tmp_path / "missing.yaml", "base.pt", epochs=1, imgsz=640, device="cpu")
    with pytest.raises(ValueError, match="workers"):
        finetune(
            tmp_path / "data.yaml",
            "base.pt",
            epochs=1,
            imgsz=640,
            device="cpu",
            workers=-1,
        )


def test_evaluation_validates_inputs_before_loading_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="imgsz"):
        evaluate_detection("base.pt", tmp_path / "data.yaml", imgsz=16)
    with pytest.raises(FileNotFoundError):
        evaluate_detection("base.pt", tmp_path / "missing.yaml", imgsz=320)

