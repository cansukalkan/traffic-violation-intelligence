"""Qwen-VL preflight with an explicit opt-in for multi-gigabyte model loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from traffic_violation.config import load_models_config
from traffic_violation.events import ViolationCandidate
from traffic_violation.vlm.parsing import parse_vlm_output
from traffic_violation.vlm.qwen import QwenDescriber, assess_qwen_readiness

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "08_qwen.txt"


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    cfg = load_models_config(ROOT / "configs" / "models.yaml")
    readiness = assess_qwen_readiness(cfg.vlm_4bit)
    lines = [
        "DEMO 08 — Qwen3-VL donanım ve adaptör kontrolü",
        f"Model: {cfg.vlm_id}",
        f"4-bit: {cfg.vlm_4bit}",
        f"Hazır: {readiness.ready}",
        f"Cihaz: {readiness.device}",
        f"Neden: {readiness.reason}",
    ]
    if args.run_model:
        if not readiness.ready:
            raise RuntimeError(readiness.reason)
        if args.image is None:
            raise ValueError("--run-model requires --image")
        candidate = ViolationCandidate("red_light", 7, 0.8, "car", {"signal": "red"})
        raw = QwenDescriber(
            cfg.vlm_id,
            max_tokens=cfg.vlm_max_tokens,
            use_4bit=cfg.vlm_4bit,
        ).describe(_read_image(args.image), candidate)
        report = parse_vlm_output(raw)
        lines.extend(
            [
                f"Model çıktısı: {raw}",
                f"Ayrıştırılmış güven: {report.confidence:.3f}",
            ]
        )
    else:
        lines.append("Model yükleme: ATLANDI (yalnızca --run-model ile etkinleşir)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Rapor: {OUTPUT}")


if __name__ == "__main__":
    main()
