"""Build the self-contained final HTML portfolio demo."""

from pathlib import Path

from traffic_violation.reporting.portfolio import build_portfolio_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "15_full_demo.html"


def main() -> None:
    result = build_portfolio_report(ROOT, OUTPUT)
    print(f"Tam portföy demosu: {result}")


if __name__ == "__main__":
    main()
