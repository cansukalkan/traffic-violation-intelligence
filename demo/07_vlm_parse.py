"""Offline demonstration of the VLM contract and parsing boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_violation.events import ViolationCandidate
from traffic_violation.vlm.fake import FakeDescriber
from traffic_violation.vlm.parsing import parse_vlm_output

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "07_vlm_parse.txt"


def main() -> None:
    candidate = ViolationCandidate(
        kind="red_light",
        track_id=7,
        frame_t=0.8,
        cls="car",
        evidence={"line": "main", "signal": "red"},
    )
    raw = """Model note:
```json
{"violation_confirmed": true, "violation_type": "red_light",
 "confidence": 0.91, "explanation": "Araç kırmızı ışıkta çizgiyi geçti.",
 "visible_evidence": ["kırmızı sinyal", "çizginin ötesindeki araç"]}
```"""
    describer = FakeDescriber([raw])
    response = describer.describe(np.zeros((32, 32, 3), dtype=np.uint8), candidate)
    report = parse_vlm_output(response)
    output = "\n".join(
        [
            "DEMO 07 — VLM güvenli ayrıştırma",
            "Sağlayıcı: FakeDescriber (çevrimdışı ve deterministik)",
            f"Onay: {report.violation_confirmed}",
            f"Tür: {report.violation_type}",
            f"Güven: {report.confidence:.2f}",
            f"Açıklama: {report.explanation}",
            f"Görsel kanıt: {', '.join(report.visible_evidence)}",
            "Karar niteliği: inceleme adayı; otomatik ceza kararı değildir.",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"Rapor: {OUTPUT}")


if __name__ == "__main__":
    main()
