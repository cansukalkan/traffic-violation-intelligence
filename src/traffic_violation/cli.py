"""Command-line interface for ingestion checks, benchmarks, and evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from traffic_violation.config import load_site_config
from traffic_violation.detection.fake import FakeDetector
from traffic_violation.eval.violation_metrics import match_events, precision_recall_f1
from traffic_violation.optimize.benchmark import benchmark_detector
from traffic_violation.pipeline.frame_source import frame_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tv", description="Traffic violation pipeline tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="validate a source and pipeline config")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--source", type=Path)
    run_parser.add_argument("--max-frames", type=int, default=100)

    bench_parser = subparsers.add_parser("bench", help="run a detector protocol benchmark")
    bench_parser.add_argument("--config", required=True, type=Path)
    bench_parser.add_argument("--frames", type=int, default=100)
    bench_parser.add_argument("--warmup", type=int, default=10)

    eval_parser = subparsers.add_parser("eval", help="compare predicted and labeled events")
    eval_parser.add_argument("--config", required=True, type=Path)
    eval_parser.add_argument("--pred", type=Path)
    eval_parser.add_argument("--gold", type=Path)
    eval_parser.add_argument("--tolerance", type=float, default=0.5)
    return parser


def _event_records(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("events")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"event file must contain a list or an events list: {path}")
    return [dict(item) for item in raw]


def _run_command(args: argparse.Namespace) -> int:
    config = load_site_config(args.config)
    if args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.source is None:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "camera_id": config.camera_id,
                    "message": "configuration valid; provide --source for ingestion check",
                },
                ensure_ascii=False,
            )
        )
        return 0
    count = 0
    last_timestamp = 0.0
    for timestamp, _ in frame_source(args.source):
        count += 1
        last_timestamp = timestamp
        if count >= args.max_frames:
            break
    print(
        json.dumps(
            {
                "status": "source-ok",
                "camera_id": config.camera_id,
                "frames_read": count,
                "last_timestamp_s": last_timestamp,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _bench_command(args: argparse.Namespace) -> int:
    load_site_config(args.config)
    if args.frames < 1:
        raise ValueError("frames must be positive")
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(args.frames)]
    detector = FakeDetector([[]] * (args.frames + max(args.warmup, 0)))
    result = benchmark_detector(detector, frames, warmup=args.warmup)
    print(
        json.dumps(
            {
                "mode": "protocol-reference",
                "fps": result.fps,
                "p50_ms": result.p50_ms,
                "p95_ms": result.p95_ms,
                "peak_mem_mb": result.peak_mem_mb,
                "measured_frames": result.measured_frames,
            },
            sort_keys=True,
        )
    )
    return 0


def _eval_command(args: argparse.Namespace) -> int:
    load_site_config(args.config)
    if args.pred is None or args.gold is None:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "message": "provide --pred and --gold event JSON files",
                }
            )
        )
        return 0
    result = match_events(
        _event_records(args.pred),
        _event_records(args.gold),
        tolerance_s=args.tolerance,
    )
    print(json.dumps(precision_recall_f1(result), sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"run": _run_command, "bench": _bench_command, "eval": _eval_command}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
