"""Object-detection contracts and implementations."""

from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.types import Detection, Detector, filter_classes
from traffic_violation.detection.yolo import YoloDetector

__all__ = ["Detection", "Detector", "FakeDetector", "YoloDetector", "filter_classes"]

