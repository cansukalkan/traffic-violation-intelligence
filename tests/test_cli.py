import json
from pathlib import Path

import pytest

from traffic_violation.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_commands_parse() -> None:
    parser = build_parser()
    for command in ("run", "bench", "eval"):
        args = parser.parse_args([command, "--config", "configs/site.yaml"])
        assert args.command == command


def test_cli_run_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", str(ROOT / "configs" / "site.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
