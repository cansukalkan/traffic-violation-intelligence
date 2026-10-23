"""Sprint 0 configuration summary demo."""

from __future__ import annotations

from pathlib import Path

from traffic_violation.config import (
    load_hardware_config,
    load_models_config,
    load_site_config,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "01_config.txt"


def main() -> None:
    site = load_site_config(ROOT / "configs" / "site.yaml")
    models = load_models_config(ROOT / "configs" / "models.yaml")
    hardware = load_hardware_config(ROOT / "configs" / "hardware.yaml")
    output = "\n".join(
        [
            "DEMO 01 — Konfigürasyon özeti",
            f"Kamera: {site.camera_id}",
            f"Bölgeler: {len(site.zones)}; durak çizgileri: {len(site.stop_lines)}",
            f"Ajan işçisi toplamı: {sum(site.pools.values())}",
            f"YOLO: {models.yolo_weights}",
            f"VLM: {models.vlm_id} (4-bit={models.vlm_4bit})",
            f"GPU: {hardware.gpu_name or 'doğrulanmadı'}; VRAM={hardware.vram_gb:g} GB",
            f"Hız yolları: {', '.join(hardware.paths)}",
        ]
    ) + "\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()

