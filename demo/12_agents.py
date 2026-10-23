"""Run the 12 logical roles with 31 workers and report backpressure metrics."""

from __future__ import annotations

import asyncio
from pathlib import Path

from traffic_violation.agents.orchestrator import run_role_pipeline
from traffic_violation.agents.roles import build_role_specs
from traffic_violation.config import load_site_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "12_agents.txt"


async def _run() -> str:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    roles = build_role_specs(cfg.pools)

    async def worker(role: str, item: int) -> int:
        delay = 0.0002 if role in {"Detector", "Analyst"} else 0.00005
        await asyncio.sleep(delay)
        return item

    result = await run_role_pipeline(range(60), roles, worker, queue_size=64)
    pressure_result = await run_role_pipeline(range(60), roles, worker, queue_size=20)
    lines = [
        "DEMO 12 — 31 işçili ajan katmanı",
        f"Mantıksal rol: {len(roles)}",
        f"Toplam işçi: {sum(role.workers for role in roles)}",
        "role | workers | processed | dropped | max_q | p50_ms | p95_ms",
    ]
    lines.extend(
        f"{metric.role} | {metric.workers} | {metric.processed} | {metric.dropped} | "
        f"{metric.max_queue_depth} | {metric.p50_ms:.3f} | {metric.p95_ms:.3f}"
        for metric in result.metrics
    )
    lines.extend(
        [
            f"Normal koşu: 60 giriş; {len(result.outputs)} tamamlanan; "
            f"{result.metrics[0].dropped} düşürülen",
            f"Basınç koşusu (kuyruk=20): 60 giriş; "
            f"{len(pressure_result.outputs)} tamamlanan; "
            f"{pressure_result.metrics[0].dropped} kontrollü düşürülen",
            "Tek kamera referans topolojisi: 31 işçi",
            "Çoklu kamera işçi sayısı, Sprint 7 benchmark sonucuna göre belirlenecek.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output = asyncio.run(_run())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"Rapor: {OUTPUT}")


if __name__ == "__main__":
    main()
