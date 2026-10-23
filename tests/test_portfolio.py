from pathlib import Path

from traffic_violation.reporting.portfolio import build_portfolio_report


def test_portfolio_report_is_self_contained(tmp_path: Path) -> None:
    root = tmp_path / "project"
    demos = root / "reports" / "demos"
    demos.mkdir(parents=True)
    for name in (
        "04_detect.jpg",
        "03_signal_preview.png",
        "05_tracking_preview.png",
        "11_pipeline_preview.png",
        "13_bench.png",
        "14_eval.png",
    ):
        (demos / name).write_bytes(b"image")
    for name in ("13_bench.txt", "14_eval.txt", "12_agents.txt"):
        (demos / name).write_text("verified <value>", encoding="utf-8")

    output = build_portfolio_report(root, tmp_path / "report.html")
    content = output.read_text(encoding="utf-8")
    assert content.count("data:image/") == 6
    assert "verified &lt;value&gt;" in content
    assert "reports/demos" not in content
