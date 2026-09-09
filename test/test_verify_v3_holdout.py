import json

import pytest

from scripts.verify_v3_holdout import build_report


def _write_capture(tmp_path, *, validation_start=20.0):
    manifest = {
        "phases": [
            {
                "name": "d_optimal_train",
                "split": "train",
                "fit_eligible": True,
                "excluded_from_fit": False,
            },
            {
                "name": "d_optimal_validation",
                "split": "validation",
                "fit_eligible": False,
                "excluded_from_fit": True,
            },
        ]
    }
    events = [
        {"phase": "d_optimal_train", "event": "start", "split": "train", "ros_time_sec": 1.0},
        {"phase": "d_optimal_train", "event": "end", "split": "train", "ros_time_sec": 11.0},
        {
            "phase": "d_optimal_validation",
            "event": "start",
            "split": "validation",
            "ros_time_sec": validation_start,
        },
        {
            "phase": "d_optimal_validation",
            "event": "end",
            "split": "validation",
            "ros_time_sec": 30.0,
        },
    ]
    (tmp_path / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "phase_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )


def test_build_report_keeps_validation_out_of_fit_and_excludes_gap(tmp_path):
    _write_capture(tmp_path)

    report = build_report(tmp_path)

    assert report["valid"] is True
    assert [chunk["name"] for chunk in report["training_chunks"]] == ["d_optimal_train"]
    assert report["validation_chunks"][0]["excluded_from_fit"] is True
    assert report["excluded_interphase_content"] == [
        {
            "start": 11.0,
            "end": 20.0,
            "duration_sec": 9.0,
            "content": "unassigned transition/reposition/hold; excluded from fitting and validation",
        }
    ]


def test_build_report_rejects_overlapping_train_and_validation(tmp_path):
    _write_capture(tmp_path, validation_start=10.0)

    with pytest.raises(ValueError, match="overlap"):
        build_report(tmp_path)
