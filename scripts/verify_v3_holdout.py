#!/usr/bin/env python3
"""Verify that a v3 capture contains a real, non-overlapping holdout segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_events(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        events.append(payload)
    return events


def build_report(run_dir: Path) -> dict:
    manifest_path = run_dir / "collection_manifest.json"
    events_path = run_dir / "phase_events.jsonl"
    manifest = _load_json(manifest_path)
    events = _load_events(events_path)
    phases = {
        str(item["name"]): item
        for item in manifest.get("phases", [])
        if isinstance(item, dict) and item.get("name")
    }

    active: dict[str, dict] = {}
    chunks = []
    for event in events:
        name = str(event.get("phase", "")).strip()
        state = str(event.get("event", "")).strip().lower()
        if not name or state not in ("start", "end"):
            continue
        if "ros_time_sec" not in event:
            raise ValueError(f"{name} {state} event has no ros_time_sec")
        if state == "start":
            if name in active:
                raise ValueError(f"{name} has a duplicate start event")
            active[name] = event
            continue
        start_event = active.pop(name, None)
        if start_event is None:
            raise ValueError(f"{name} has an end event without a start")
        start = float(start_event["ros_time_sec"])
        end = float(event["ros_time_sec"])
        if end <= start:
            raise ValueError(f"{name} has a non-positive interval [{start}, {end}]")
        phase = phases.get(name, {})
        split = str(event.get("split") or start_event.get("split") or phase.get("split") or "train").lower()
        role = "validation" if split.startswith("val") else "train"
        fit_eligible = role == "train"
        if "fit_eligible" in phase and bool(phase["fit_eligible"]) != fit_eligible:
            raise ValueError(f"{name} has inconsistent fit_eligible metadata")
        if "excluded_from_fit" in phase and bool(phase["excluded_from_fit"]) == fit_eligible:
            raise ValueError(f"{name} has inconsistent excluded_from_fit metadata")
        chunks.append(
            {
                "name": name,
                "role": role,
                "start": start,
                "end": end,
                "duration_sec": end - start,
                "fit_eligible": fit_eligible,
                "excluded_from_fit": not fit_eligible,
            }
        )

    if active:
        raise ValueError(f"Missing end event for: {', '.join(sorted(active))}")
    chunks.sort(key=lambda item: item["start"])
    train_chunks = [chunk for chunk in chunks if chunk["role"] == "train"]
    validation_chunks = [chunk for chunk in chunks if chunk["role"] == "validation"]
    if not train_chunks:
        raise ValueError("No completed training chunk was found")
    if not validation_chunks:
        raise ValueError("No completed validation holdout chunk was found")

    excluded_intervals = []
    for left, right in zip(chunks, chunks[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"Chunks {left['name']} and {right['name']} overlap")
        if right["start"] > left["end"]:
            excluded_intervals.append(
                {
                    "start": left["end"],
                    "end": right["start"],
                    "duration_sec": right["start"] - left["end"],
                    "content": "unassigned transition/reposition/hold; excluded from fitting and validation",
                }
            )

    return {
        "schema": "franka_sysid_v3_holdout_report_v1",
        "run_dir": str(run_dir),
        "valid": True,
        "training_chunks": train_chunks,
        "validation_chunks": validation_chunks,
        "excluded_interphase_content": excluded_intervals,
        "fit_policy": "only role=train chunks are fit-eligible; validation and unassigned samples are excluded",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report = build_report(run_dir)
    output = args.output.expanduser().resolve() if args.output else run_dir / "holdout_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    train_seconds = sum(chunk["duration_sec"] for chunk in report["training_chunks"])
    validation_seconds = sum(chunk["duration_sec"] for chunk in report["validation_chunks"])
    print(
        f"Valid holdout: {len(report['training_chunks'])} train chunk(s), "
        f"{len(report['validation_chunks'])} validation chunk(s), "
        f"{train_seconds:.3f}s train / {validation_seconds:.3f}s validation"
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
