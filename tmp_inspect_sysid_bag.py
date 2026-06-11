#!/usr/bin/env python3
"""Inspect Franka SysID rosbag2 data without a ROS installation.

This decodes just enough ROS 2 CDR to check whether collected bags contain
joint effort/torque readings and makes slide-ready joint plots. It supports
rosbag2 bags stored as MCAP or SQLite3 (.db3). MCAP reading requires:

    python -m pip install mcap

Plotting requires:

    python -m pip install matplotlib

Example:

    python tmp_inspect_sysid_bag.py /path/to/franka_v2_001/bag
    python tmp_inspect_sysid_bag.py /path/to/franka_v2_001/bag --csv efforts.csv
    python tmp_inspect_sysid_bag.py /path/to/franka_v2_001/bag --plots-dir slide_plots
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class CdrError(ValueError):
    pass


class CdrReader:
    """Tiny ROS 2 CDR reader for the message types used by this collector."""

    def __init__(self, data: bytes, *, alignment_origin: int = 0):
        if len(data) < 4:
            raise CdrError("message is too short for a CDR encapsulation header")

        # ROS 2 bags normally use CDR little-endian encapsulation 00 01 00 00.
        enc0, enc1 = data[0], data[1]
        self.endian = "<" if enc1 in (1, 3) or enc0 in (1, 3) else ">"
        self.data = memoryview(data)
        self.pos = 4
        self.alignment_origin = alignment_origin

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def align(self, alignment: int) -> None:
        rel = self.pos - self.alignment_origin
        pad = (alignment - (rel % alignment)) % alignment
        self.pos += pad
        if self.pos > len(self.data):
            raise CdrError("alignment advanced past end of message")

    def _take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise CdrError(f"wanted {size} bytes at offset {self.pos}, message has {len(self.data)} bytes")
        out = self.data[self.pos : self.pos + size].tobytes()
        self.pos += size
        return out

    def read_int32(self) -> int:
        self.align(4)
        value = struct.unpack_from(self.endian + "i", self.data, self.pos)[0]
        self.pos += 4
        return int(value)

    def read_uint32(self) -> int:
        self.align(4)
        value = struct.unpack_from(self.endian + "I", self.data, self.pos)[0]
        self.pos += 4
        return int(value)

    def read_double(self) -> float:
        self.align(8)
        value = struct.unpack_from(self.endian + "d", self.data, self.pos)[0]
        self.pos += 8
        return float(value)

    def read_string(self) -> str:
        length = self.read_uint32()
        if length > self.remaining() or length > 1_000_000:
            raise CdrError(f"implausible string length {length}")
        raw = self._take(length)
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def read_string_seq(self) -> list[str]:
        count = self.read_uint32()
        if count > 10_000:
            raise CdrError(f"implausible string sequence length {count}")
        return [self.read_string() for _ in range(count)]

    def read_double_seq(self) -> list[float]:
        count = self.read_uint32()
        if count > 100_000:
            raise CdrError(f"implausible float64 sequence length {count}")
        self.align(8)
        values = []
        for _ in range(count):
            if self.remaining() < 8:
                raise CdrError("float64 sequence overruns message")
            values.append(struct.unpack_from(self.endian + "d", self.data, self.pos)[0])
            self.pos += 8
        return [float(v) for v in values]

    def read_time_ns(self) -> int:
        sec = self.read_int32()
        nsec = self.read_uint32()
        return int(sec) * 1_000_000_000 + int(nsec)

    def read_header(self) -> dict[str, Any]:
        return {"stamp_ns": self.read_time_ns(), "frame_id": self.read_string()}

    def read_joint_trajectory_point(self) -> dict[str, Any]:
        return {
            "positions": self.read_double_seq(),
            "velocities": self.read_double_seq(),
            "accelerations": self.read_double_seq(),
            "effort": self.read_double_seq(),
            "time_from_start_ns": self.read_time_ns(),
        }


def _decode_with_alignment_fallback(data: bytes, decode_func):
    errors = []
    for origin in (0, 4):
        try:
            return decode_func(CdrReader(data, alignment_origin=origin))
        except Exception as exc:  # noqa: BLE001 - preserve both parse attempts.
            errors.append(f"origin={origin}: {exc}")
    raise CdrError("; ".join(errors))


def decode_joint_state(data: bytes) -> dict[str, Any]:
    def decode(reader: CdrReader) -> dict[str, Any]:
        header = reader.read_header()
        names = reader.read_string_seq()
        position = reader.read_double_seq()
        velocity = reader.read_double_seq()
        effort = reader.read_double_seq()
        if len(names) > 200:
            raise CdrError(f"implausible joint count {len(names)}")
        for field_name, values in (("position", position), ("velocity", velocity), ("effort", effort)):
            if values and len(values) != len(names):
                raise CdrError(f"{field_name} length {len(values)} does not match names length {len(names)}")
        return {"header": header, "name": names, "position": position, "velocity": velocity, "effort": effort}

    return _decode_with_alignment_fallback(data, decode)


def joint_point_field_names(schema_text: str | None) -> list[str]:
    if not schema_text:
        return []
    text = schema_text.split("================================================================================")[0]
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        msg_type, field_name = parts[0], parts[1].split("=", 1)[0]
        type_name = msg_type.split("/")[-1]
        if type_name == "JointTrajectoryPoint":
            names.append(field_name)
    return names


def decode_controller_state(data: bytes, schema_text: str | None = None) -> dict[str, Any]:
    schema_fields = joint_point_field_names(schema_text)
    fallback_fields = ["reference", "feedback", "error", "output"]
    field_names = schema_fields or fallback_fields
    expected_points = len(schema_fields) if schema_fields else len(fallback_fields)

    def decode(reader: CdrReader) -> dict[str, Any]:
        header = reader.read_header()
        joint_names = reader.read_string_seq()
        if not 0 < len(joint_names) <= 200:
            raise CdrError(f"implausible controller joint count {len(joint_names)}")

        points: dict[str, Any] = {}
        for point_index in range(expected_points):
            if reader.remaining() <= 0:
                break
            field_name = field_names[point_index] if point_index < len(field_names) else f"point_{point_index}"
            try:
                points[field_name] = reader.read_joint_trajectory_point()
            except CdrError:
                if schema_fields or len(points) < 2:
                    raise
                # Without schema text, tolerate trailing fields such as
                # speed_scaling_factor after the standard trajectory points.
                break

        if len(points) < 2:
            raise CdrError("controller state did not contain enough trajectory points")
        return {"header": header, "joint_names": joint_names, "points": points}

    return _decode_with_alignment_fallback(data, decode)


@dataclass
class TopicInfo:
    topic_type: str = ""
    count: int = 0
    first_ns: int | None = None
    last_ns: int | None = None

    def update(self, timestamp_ns: int) -> None:
        self.count += 1
        self.first_ns = timestamp_ns if self.first_ns is None else min(self.first_ns, timestamp_ns)
        self.last_ns = timestamp_ns if self.last_ns is None else max(self.last_ns, timestamp_ns)


class EffortStats:
    def __init__(self, label: str):
        self.label = label
        self.message_count = 0
        self.with_effort_count = 0
        self.empty_effort_count = 0
        self.all_zero_count = 0
        self.decode_errors = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.joint_names: list[str] = []
        self._n: list[int] = []
        self._nonzero: list[int] = []
        self._sum: list[float] = []
        self._sum_abs: list[float] = []
        self._sum_sq: list[float] = []
        self._min: list[float] = []
        self._max: list[float] = []

    def note_message(self, timestamp_ns: int) -> None:
        self.message_count += 1
        self.first_ns = timestamp_ns if self.first_ns is None else min(self.first_ns, timestamp_ns)
        self.last_ns = timestamp_ns if self.last_ns is None else max(self.last_ns, timestamp_ns)

    def note_decode_error(self) -> None:
        self.decode_errors += 1

    def update(self, timestamp_ns: int, joint_names: list[str], effort: list[float]) -> None:
        self.note_message(timestamp_ns)
        if not effort:
            self.empty_effort_count += 1
            return

        self.with_effort_count += 1
        if all(abs(value) <= 1e-9 for value in effort if math.isfinite(value)):
            self.all_zero_count += 1

        if len(joint_names) > len(self.joint_names):
            self.joint_names = list(joint_names)
        self._ensure_len(len(effort))
        for i, value in enumerate(effort):
            if not math.isfinite(value):
                continue
            self._n[i] += 1
            self._sum[i] += value
            self._sum_abs[i] += abs(value)
            self._sum_sq[i] += value * value
            self._min[i] = min(self._min[i], value)
            self._max[i] = max(self._max[i], value)
            if abs(value) > 1e-9:
                self._nonzero[i] += 1

    def _ensure_len(self, length: int) -> None:
        while len(self._n) < length:
            self._n.append(0)
            self._nonzero.append(0)
            self._sum.append(0.0)
            self._sum_abs.append(0.0)
            self._sum_sq.append(0.0)
            self._min.append(float("inf"))
            self._max.append(float("-inf"))

    def duration_sec(self) -> float:
        if self.first_ns is None or self.last_ns is None:
            return 0.0
        return max(0.0, (self.last_ns - self.first_ns) / 1e9)

    def has_nonzero_effort(self) -> bool:
        return any(count > 0 for count in self._nonzero)

    def print_report(self) -> None:
        print(f"\n{self.label}")
        print("-" * len(self.label))
        print(f"messages decoded: {self.message_count}")
        print(f"time span: {self.duration_sec():.3f} s")
        print(f"messages with effort arrays: {self.with_effort_count}")
        print(f"messages with empty effort arrays: {self.empty_effort_count}")
        print(f"messages where effort is all zero: {self.all_zero_count}")
        if self.decode_errors:
            print(f"decode errors: {self.decode_errors}")
        if not self.with_effort_count:
            print("No effort samples found.")
            return

        print("per-joint effort stats, units are whatever the source JointState used, normally Nm:")
        print("  joint                 n        min        max   mean_abs        rms  nonzero%")
        for i, n in enumerate(self._n):
            if n <= 0:
                continue
            name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
            mean_abs = self._sum_abs[i] / n
            rms = math.sqrt(self._sum_sq[i] / n)
            nonzero_pct = 100.0 * self._nonzero[i] / n
            print(
                f"  {name:<16} {n:7d} {self._min[i]:10.4g} {self._max[i]:10.4g}"
                f" {mean_abs:10.4g} {rms:10.4g} {nonzero_pct:8.2f}"
            )


class JointTimeSeries:
    def __init__(self, label: str):
        self.label = label
        self.timestamps_ns: list[int] = []
        self.joint_names: list[str] = []
        self.position: list[list[float]] = []
        self.velocity: list[list[float]] = []
        self.effort: list[list[float]] = []

    def update(
        self,
        timestamp_ns: int,
        joint_names: list[str],
        position: list[float],
        velocity: list[float],
        effort: list[float],
        requested_joints: list[str] | None,
    ) -> None:
        names, selected_position = select_joint_values(joint_names, position, requested_joints)
        _, selected_velocity = select_joint_values(joint_names, velocity, names)
        _, selected_effort = select_joint_values(joint_names, effort, names)
        if not names or not (selected_position or selected_velocity or selected_effort):
            return

        width = len(names)
        if not self.joint_names:
            self.joint_names = list(names)
        elif self.joint_names != names:
            # Keep plots rectangular. This collector should be stable, but this
            # avoids mixing different joint layouts into one figure.
            return

        self.timestamps_ns.append(timestamp_ns)
        self.position.append(normalize_series_row(selected_position, width))
        self.velocity.append(normalize_series_row(selected_velocity, width))
        self.effort.append(normalize_series_row(selected_effort, width))

    def has_samples(self) -> bool:
        return bool(self.timestamps_ns and self.joint_names)


def normalize_series_row(values: list[float], width: int) -> list[float]:
    row = [float("nan")] * width
    for i, value in enumerate(values[:width]):
        row[i] = float(value)
    return row


def select_joint_values(
    joint_names: list[str],
    values: list[float],
    requested_joints: list[str] | None,
) -> tuple[list[str], list[float]]:
    if not values:
        return (list(requested_joints or joint_names), [])
    if requested_joints is None:
        width = min(len(joint_names), len(values))
        return list(joint_names[:width]), [float(value) for value in values[:width]]

    index = {name: i for i, name in enumerate(joint_names)}
    selected_names: list[str] = []
    selected_values: list[float] = []
    for name in requested_joints:
        i = index.get(name)
        if i is None or i >= len(values):
            continue
        selected_names.append(name)
        selected_values.append(float(values[i]))
    return selected_names, selected_values


def default_plots_dir(bag_path: Path) -> Path:
    if bag_path.is_file():
        return bag_path.parent / "sysid_plots"
    if bag_path.name == "bag":
        return bag_path.parent / "sysid_plots"
    return bag_path / "sysid_plots"


def plot_joint_series(series: JointTimeSeries, output_dir: Path) -> list[Path]:
    if not series.has_samples():
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requested, but Python package 'matplotlib' is not installed.\n"
            "Install it on the non-ROS machine with: python -m pip install matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = [(stamp - series.timestamps_ns[0]) / 1e9 for stamp in series.timestamps_ns]
    plot_specs = [
        ("Joint position", "Position (rad)", series.position),
        ("Joint velocity", "Velocity (rad/s)", series.velocity),
        ("Joint torque", "Torque / effort (Nm)", series.effort),
    ]

    path = output_dir / "joint_collection_overview.png"
    fig, axes = plt.subplots(3, 1, figsize=(13.333, 7.5), dpi=144, sharex=True)
    fig.patch.set_facecolor("white")
    colors = plt.get_cmap("tab10").colors
    legend_handles = []
    legend_labels = []

    for ax, (title, ylabel, rows) in zip(axes, plot_specs):
        ax.set_facecolor("white")
        has_data = rows and not all(all(not math.isfinite(value) for value in row) for row in rows)
        if has_data:
            for joint_i, joint_name in enumerate(series.joint_names):
                y = [row[joint_i] if joint_i < len(row) else float("nan") for row in rows]
                (line,) = ax.plot(
                    time_s,
                    y,
                    linewidth=1.35,
                    color=colors[joint_i % len(colors)],
                    label=joint_name,
                )
                if len(legend_handles) < len(series.joint_names):
                    legend_handles.append(line)
                    legend_labels.append(joint_name)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="#6b7280",
            )

        ax.text(
            0.01,
            0.88,
            title,
            transform=ax.transAxes,
            fontsize=13,
            weight="semibold",
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, color="#d8dee9", linewidth=0.8, alpha=0.85)
        ax.tick_params(axis="both", labelsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#4b5563")
        ax.spines["bottom"].set_color("#4b5563")

    axes[-1].set_xlabel("Time since start (s)", fontsize=14)
    fig.suptitle("Franka Joint Data Over Collection", fontsize=22, weight="semibold", y=0.98)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(7, max(1, len(series.joint_names))),
            frameon=False,
            fontsize=10.5,
        )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.13, hspace=0.20)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [path]


def selected_effort(
    joint_names: list[str],
    effort: list[float],
    requested_joints: list[str] | None,
) -> tuple[list[str], list[float], list[str]]:
    if not effort:
        return list(joint_names), [], []
    if not requested_joints:
        return list(joint_names[: len(effort)]), list(effort), []

    index = {name: i for i, name in enumerate(joint_names)}
    missing = [name for name in requested_joints if name not in index]
    if missing:
        return list(joint_names[: len(effort)]), list(effort), missing

    selected = []
    for name in requested_joints:
        i = index[name]
        selected.append(float(effort[i]) if i < len(effort) else float("nan"))
    return list(requested_joints), selected, []


def measured_point(points: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in ("feedback", "actual", "measured"):
        if key in points:
            return key, points[key]
    items = list(points.items())
    if len(items) >= 2:
        return items[1]
    return items[0]


def iter_mcap_messages(paths: Iterable[Path]):
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise SystemExit(
            "MCAP bag found, but Python package 'mcap' is not installed.\n"
            "Install it on the non-ROS machine with: python -m pip install mcap"
        ) from exc

    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages():
                schema_text = None
                schema_name = ""
                if schema is not None:
                    schema_name = schema.name or ""
                    if schema.data:
                        schema_text = schema.data.decode("utf-8", errors="replace")
                yield {
                    "file": path,
                    "topic": channel.topic,
                    "type": schema_name,
                    "encoding": channel.message_encoding,
                    "timestamp_ns": int(message.log_time),
                    "data": bytes(message.data),
                    "schema_text": schema_text,
                }


def iter_db3_messages(paths: Iterable[Path]):
    for path in paths:
        conn = sqlite3.connect(str(path))
        try:
            topics = {
                int(row[0]): {"topic": str(row[1]), "type": str(row[2])}
                for row in conn.execute("SELECT id, name, type FROM topics")
            }
            for topic_id, timestamp_ns, data in conn.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
            ):
                topic = topics.get(int(topic_id), {"topic": f"<topic_id:{topic_id}>", "type": ""})
                yield {
                    "file": path,
                    "topic": topic["topic"],
                    "type": topic["type"],
                    "encoding": "cdr",
                    "timestamp_ns": int(timestamp_ns),
                    "data": bytes(data),
                    "schema_text": None,
                }
        finally:
            conn.close()


def discover_bag_files(path: Path) -> tuple[list[Path], list[Path]]:
    if path.is_file():
        return ([path] if path.suffix == ".mcap" else []), ([path] if path.suffix == ".db3" else [])
    return sorted(path.rglob("*.mcap")), sorted(path.rglob("*.db3"))


def find_topic_map(path: Path) -> Path | None:
    candidates = []
    if path.is_dir():
        candidates.extend([path / "franka_sysid_topic_map.yaml", path.parent / "franka_sysid_topic_map.yaml"])
    else:
        candidates.extend([path.parent / "franka_sysid_topic_map.yaml", path.parent.parent / "franka_sysid_topic_map.yaml"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_torque_topic_from_map(path: Path) -> str | None:
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("torque_topic:"):
            return line.split(":", 1)[1].strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Bag directory, .mcap file, or .db3 file to inspect.")
    parser.add_argument("--telemetry-topic", default="/sysid/controller_state")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--joints", nargs="*", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--all-joints", action="store_true", help="Do not filter to the Panda arm joint names.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional long-form CSV output of effort samples.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for the slide-ready position/velocity/torque overview PNG. Defaults to sysid_plots next to the bag.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation.")
    parser.add_argument("--max-messages", type=int, default=0, help="Optional cap for quick inspection; 0 means all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bag_path = args.bag.expanduser().resolve()
    if not bag_path.exists():
        print(f"Bag path does not exist: {bag_path}", file=sys.stderr)
        return 2

    mcap_files, db3_files = discover_bag_files(bag_path)
    if not mcap_files and not db3_files:
        print(f"No .mcap or .db3 files found under: {bag_path}", file=sys.stderr)
        return 2

    requested_joints = None if args.all_joints else list(args.joints or [])
    telemetry_stats = EffortStats(f"{args.telemetry_topic} measured point effort")
    joint_state_stats = EffortStats(f"{args.joint_states_topic} effort")
    telemetry_series = JointTimeSeries(args.telemetry_topic)
    joint_state_series = JointTimeSeries(args.joint_states_topic)
    topics: dict[str, TopicInfo] = {}
    missing_filter_names: set[str] = set()

    csv_file = None
    csv_writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["source", "topic", "timestamp_ns", "stamp_ns", "point", "joint", "effort"])
        csv_writer.writeheader()

    def maybe_write_csv(source: str, topic: str, timestamp_ns: int, stamp_ns: int, point: str, names: list[str], effort: list[float]) -> None:
        if csv_writer is None:
            return
        for name, value in zip(names, effort):
            csv_writer.writerow(
                {
                    "source": source,
                    "topic": topic,
                    "timestamp_ns": timestamp_ns,
                    "stamp_ns": stamp_ns,
                    "point": point,
                    "joint": name,
                    "effort": value,
                }
            )

    try:
        streams = []
        if mcap_files:
            streams.append(iter_mcap_messages(mcap_files))
        if db3_files:
            streams.append(iter_db3_messages(db3_files))

        seen_messages = 0
        for stream in streams:
            for record in stream:
                seen_messages += 1
                topic = record["topic"]
                topic_type = record["type"]
                timestamp_ns = record["timestamp_ns"]
                info = topics.setdefault(topic, TopicInfo(topic_type=topic_type))
                if topic_type and not info.topic_type:
                    info.topic_type = topic_type
                info.update(timestamp_ns)

                is_telemetry = topic == args.telemetry_topic or topic_type.endswith("JointTrajectoryControllerState")
                is_joint_state = topic == args.joint_states_topic or topic_type.endswith("JointState")

                if is_telemetry:
                    try:
                        decoded = decode_controller_state(record["data"], record["schema_text"])
                        point_name, point = measured_point(decoded["points"])
                        names, effort, missing = selected_effort(
                            decoded["joint_names"], point.get("effort", []), requested_joints
                        )
                        missing_filter_names.update(missing)
                        telemetry_stats.update(timestamp_ns, names, effort)
                        maybe_write_csv(
                            "controller_state",
                            topic,
                            timestamp_ns,
                            decoded["header"]["stamp_ns"],
                            point_name,
                            names,
                            effort,
                        )
                        telemetry_series.update(
                            timestamp_ns,
                            decoded["joint_names"],
                            point.get("positions", []),
                            point.get("velocities", []),
                            point.get("effort", []),
                            requested_joints,
                        )
                    except Exception as exc:  # noqa: BLE001
                        telemetry_stats.note_decode_error()
                        if telemetry_stats.decode_errors <= 3:
                            print(f"Warning: failed to decode controller state on {topic}: {exc}", file=sys.stderr)

                if is_joint_state:
                    try:
                        decoded = decode_joint_state(record["data"])
                        names, effort, missing = selected_effort(decoded["name"], decoded["effort"], requested_joints)
                        missing_filter_names.update(missing)
                        joint_state_stats.update(timestamp_ns, names, effort)
                        maybe_write_csv(
                            "joint_state",
                            topic,
                            timestamp_ns,
                            decoded["header"]["stamp_ns"],
                            "effort",
                            names,
                            effort,
                        )
                        joint_state_series.update(
                            timestamp_ns,
                            decoded["name"],
                            decoded["position"],
                            decoded["velocity"],
                            decoded["effort"],
                            requested_joints,
                        )
                    except Exception as exc:  # noqa: BLE001
                        joint_state_stats.note_decode_error()
                        if joint_state_stats.decode_errors <= 3:
                            print(f"Warning: failed to decode JointState on {topic}: {exc}", file=sys.stderr)

                if args.max_messages and seen_messages >= args.max_messages:
                    break
            if args.max_messages and seen_messages >= args.max_messages:
                break
    finally:
        if csv_file is not None:
            csv_file.close()

    print(f"Inspected: {bag_path}")
    if mcap_files:
        print(f"MCAP files: {len(mcap_files)}")
    if db3_files:
        print(f"SQLite files: {len(db3_files)}")

    topic_map = find_topic_map(bag_path)
    if topic_map is not None:
        torque_topic = read_torque_topic_from_map(topic_map)
        print(f"Topic map: {topic_map}")
        print(f"Topic map torque_topic: {torque_topic}")

    print("\nTopic inventory")
    print("---------------")
    print("  count       span_s  topic  type")
    for topic, info in sorted(topics.items()):
        span = 0.0 if info.first_ns is None or info.last_ns is None else (info.last_ns - info.first_ns) / 1e9
        print(f"  {info.count:7d} {span:12.3f}  {topic}  {info.topic_type}")

    if missing_filter_names:
        print(
            "\nRequested joint filter did not match every message. Missing names seen: "
            + ", ".join(sorted(missing_filter_names))
        )
        print("Those messages were summarized with all joints from the message instead.")

    telemetry_stats.print_report()
    joint_state_stats.print_report()

    print("\nVerdict")
    print("-------")
    if telemetry_stats.with_effort_count and telemetry_stats.has_nonzero_effort():
        print("Telemetry contains non-zero effort arrays in the measured controller-state point.")
        print("For this collector, those values came from the configured JointState.effort source topic.")
    elif telemetry_stats.with_effort_count:
        print("Telemetry has effort arrays, but they are all zero. That usually means torques were not captured.")
    else:
        print("Telemetry does not contain usable effort arrays.")

    if topic_map is not None and read_torque_topic_from_map(topic_map) in (None, "", "null", "~"):
        print("The generated topic map has torque_topic disabled, so Isaac SysID would ignore torques unless you edit the map.")
    if args.csv is not None:
        print(f"CSV effort samples written to: {args.csv.resolve()}")
    if not args.no_plots:
        plot_source = joint_state_series if joint_state_series.has_samples() else telemetry_series
        plots_dir = (args.plots_dir or default_plots_dir(bag_path)).expanduser().resolve()
        written = plot_joint_series(plot_source, plots_dir)
        if written:
            print("\nPlots")
            print("-----")
            print(f"Source: {plot_source.label}")
            for path in written:
                print(f"Wrote: {path}")
        else:
            print("\nNo plottable joint position/velocity/torque series found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
