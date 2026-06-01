"""Methodical Franka SysID data collection with designed joint trajectories.

V1 used sinusoidal joint-space waypoints and let MoveIt retime each segment. V2
keeps MoveIt available for safe repositioning, but collects data by executing
explicit ``FollowJointTrajectory`` references:

* warmup multi-sine
* single-joint friction sweeps at multiple peak velocities
* coupled multi-sine training excitation
* held-out coupled multi-sine validation excitation
* static pose holds for gravity / bias observations
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import RobotTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .franka_sysid_collect import (
    FRANKA_AMPLITUDES,
    FRANKA_CENTER,
    FRANKA_JOINTS,
    FRANKA_LIMITS,
    BagRecorder,
    MoveItActionClient,
    SysIdTelemetryPublisher,
    duration_to_sec,
    unique_ordered,
    write_topic_map,
)


TRAIN_PHASES = {"warmup_multisine", "friction_sweeps", "coupled_multisine_train", "coupled_multisine_fast"}
VALIDATION_PHASES = {"coupled_multisine_validation"}


@dataclass
class ExcitationPhase:
    name: str
    split: str
    purpose: str
    trajectory: JointTrajectory
    hold_after_sec: float = 0.0
    use_moveit_start: bool = True

    @property
    def duration(self) -> float:
        if not self.trajectory.points:
            return 0.0
        return duration_to_sec(self.trajectory.points[-1].time_from_start)

    @property
    def start_position(self) -> list[float]:
        return list(self.trajectory.points[0].positions)


class FollowTrajectoryClient:
    """Blocking wrapper around a joint trajectory controller action."""

    def __init__(self, node: Node, action_name: str):
        self.node = node
        self.action_name = action_name
        self.client = ActionClient(node, FollowJointTrajectory, action_name)

    def wait_for_server(self, timeout_sec: float) -> None:
        if not self.client.wait_for_server(timeout_sec=timeout_sec):
            raise RuntimeError(f"FollowJointTrajectory action server is not available: {self.action_name}")

    def _wait_future(self, future, timeout_sec: float, description: str):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {description}")
            time.sleep(0.02)
        return future.result()

    def execute(self, trajectory: JointTrajectory, timeout_sec: float) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        if hasattr(goal, "goal_time_tolerance"):
            goal.goal_time_tolerance = Duration(seconds=2.0).to_msg()

        send_future = self.client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, timeout_sec, "trajectory goal acceptance")
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"{self.action_name} rejected the trajectory goal")

        result_future = goal_handle.get_result_async()
        result_response = self._wait_future(result_future, timeout_sec, "trajectory execution result")
        result = result_response.result
        error_code = int(getattr(result, "error_code", 0))
        successful_code = int(getattr(FollowJointTrajectory.Result, "SUCCESSFUL", 0))
        if error_code != successful_code:
            error_text = getattr(result, "error_string", "")
            raise RuntimeError(f"FollowJointTrajectory failed with code {error_code}: {error_text}")


def _msg_duration(seconds: float):
    return Duration(seconds=max(0.0, float(seconds))).to_msg()


def _clamp_positions(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    for joint_i, (lo, hi) in enumerate(FRANKA_LIMITS):
        out[..., joint_i] = np.clip(out[..., joint_i], lo, hi)
    return out


def _finite_difference(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    if len(times) < 2:
        return out
    for joint_i in range(values.shape[1]):
        out[:, joint_i] = np.gradient(values[:, joint_i], times, edge_order=1)
    return out


def _time_scale_for_limits(times: np.ndarray, positions: np.ndarray, max_velocity: float, max_acceleration: float) -> np.ndarray:
    velocities = _finite_difference(times, positions)
    accelerations = _finite_difference(times, velocities)
    peak_v = float(np.max(np.abs(velocities))) if velocities.size else 0.0
    peak_a = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0
    vel_scale = peak_v / max(max_velocity, 1e-6)
    acc_scale = math.sqrt(peak_a / max(max_acceleration, 1e-6))
    scale = max(1.0, vel_scale, acc_scale)
    return times * scale


def _build_trajectory(
    joint_names: list[str],
    times: np.ndarray,
    positions: np.ndarray,
    *,
    max_velocity: float,
    max_acceleration: float,
) -> JointTrajectory:
    times = np.asarray(times, dtype=np.float64)
    positions = _clamp_positions(np.asarray(positions, dtype=np.float64))
    if times.ndim != 1 or positions.ndim != 2 or len(times) != positions.shape[0]:
        raise ValueError("times must be (T,) and positions must be (T, N)")
    if len(times) < 2:
        raise ValueError("trajectory must contain at least two points")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")

    scaled_times = _time_scale_for_limits(times, positions, max_velocity, max_acceleration)
    velocities = _finite_difference(scaled_times, positions)
    accelerations = _finite_difference(scaled_times, velocities)

    trajectory = JointTrajectory()
    trajectory.joint_names = list(joint_names)
    for t, q, dq, ddq in zip(scaled_times, positions, velocities, accelerations):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [float(v) for v in dq]
        point.accelerations = [float(v) for v in ddq]
        point.time_from_start = _msg_duration(float(t))
        trajectory.points.append(point)
    return trajectory


def _multisine_positions(
    times: np.ndarray,
    *,
    base_period: float,
    amplitude_scale: float,
    variant: str,
) -> np.ndarray:
    center = np.asarray(FRANKA_CENTER, dtype=np.float64)
    amps = amplitude_scale * np.asarray(FRANKA_AMPLITUDES, dtype=np.float64)
    phases_a = {
        "warmup": [0.0, 0.6, 1.1, 1.7, 2.1, 2.8, 3.2],
        "train": [0.0, 1.7, 0.9, 2.2, 1.1, 2.8, 0.4],
        "fast": [0.4, 2.4, 1.2, 2.9, 0.2, 1.8, 2.6],
        "validation": [1.1, 0.2, 2.7, 0.7, 2.2, 1.4, 3.0],
    }[variant]
    phases_b = {
        "warmup": [1.9, 2.5, 0.4, 2.9, 0.8, 1.2, 2.2],
        "train": [2.3, 0.4, 2.6, 1.0, 2.9, 0.8, 1.5],
        "fast": [2.9, 0.9, 2.1, 0.1, 1.4, 2.7, 0.6],
        "validation": [2.8, 1.6, 0.3, 2.1, 0.9, 2.6, 1.2],
    }[variant]
    harmonics_a = {
        "warmup": [1, 1, 2, 1, 2, 1, 2],
        "train": [1, 2, 3, 2, 4, 3, 5],
        "fast": [2, 3, 4, 3, 5, 4, 6],
        "validation": [2, 1, 4, 3, 5, 2, 6],
    }[variant]
    harmonics_b = {
        "warmup": [2, 2, 3, 2, 3, 2, 3],
        "train": [3, 4, 5, 3, 6, 5, 7],
        "fast": [4, 5, 6, 5, 7, 6, 8],
        "validation": [5, 3, 6, 4, 7, 5, 8],
    }[variant]

    theta = 2.0 * math.pi * times / float(base_period)
    positions = np.zeros((len(times), len(FRANKA_JOINTS)), dtype=np.float64)
    for joint_i in range(len(FRANKA_JOINTS)):
        primary = np.sin(harmonics_a[joint_i] * theta + phases_a[joint_i])
        secondary = np.sin(harmonics_b[joint_i] * theta + phases_b[joint_i])
        positions[:, joint_i] = center[joint_i] + amps[joint_i] * (0.70 * primary + 0.30 * secondary)

    # Smoothly enter and exit the excitation envelope.
    if len(times) > 4:
        ramp = np.minimum(1.0, np.minimum(times / base_period, (times[-1] - times) / base_period))
        ramp = np.clip(ramp, 0.0, 1.0)
        positions = center + (positions - center) * ramp[:, np.newaxis]
    return _clamp_positions(positions)


def _make_multisine_phase(
    *,
    name: str,
    split: str,
    purpose: str,
    joint_names: list[str],
    cycles: int,
    sample_rate: float,
    base_period: float,
    amplitude_scale: float,
    variant: str,
    max_velocity: float,
    max_acceleration: float,
    hold_after_sec: float,
) -> ExcitationPhase:
    duration = max(base_period, float(cycles) * base_period)
    sample_count = max(3, int(round(duration * sample_rate)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    positions = _multisine_positions(times, base_period=base_period, amplitude_scale=amplitude_scale, variant=variant)
    trajectory = _build_trajectory(
        joint_names,
        times,
        positions,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )
    return ExcitationPhase(name, split, purpose, trajectory, hold_after_sec=hold_after_sec)


def _single_joint_sweep_positions(
    *,
    sample_rate: float,
    amplitude_scale: float,
    peak_velocity: float,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(FRANKA_CENTER, dtype=np.float64)
    segments: list[np.ndarray] = []
    segment_times: list[np.ndarray] = []
    t_offset = 0.0
    for joint_i, amp_nominal in enumerate(FRANKA_AMPLITUDES):
        amplitude = max(0.02, float(amplitude_scale) * float(amp_nominal))
        period = max(4.0, 2.0 * math.pi * amplitude / max(float(peak_velocity), 1e-3))
        for _repeat in range(max(1, repeats)):
            count = max(5, int(round(period * sample_rate)) + 1)
            local_t = np.linspace(0.0, period, count)
            q = np.tile(center, (count, 1))
            q[:, joint_i] = center[joint_i] + amplitude * np.sin(2.0 * math.pi * local_t / period)
            if segments:
                local_t = local_t[1:]
                q = q[1:]
            segments.append(q)
            segment_times.append(local_t + t_offset)
            t_offset += period
    return np.concatenate(segment_times), _clamp_positions(np.vstack(segments))


def _make_friction_sweep_phase(
    *,
    joint_names: list[str],
    sample_rate: float,
    amplitude_scale: float,
    peak_velocity: float,
    repeats: int,
    max_acceleration: float,
    hold_after_sec: float,
) -> ExcitationPhase:
    times, positions = _single_joint_sweep_positions(
        sample_rate=sample_rate,
        amplitude_scale=amplitude_scale,
        peak_velocity=peak_velocity,
        repeats=repeats,
    )
    trajectory = _build_trajectory(
        joint_names,
        times,
        positions,
        max_velocity=max(peak_velocity * 1.05, 0.01),
        max_acceleration=max_acceleration,
    )
    return ExcitationPhase(
        name=f"friction_sweeps_{peak_velocity:.2f}radps".replace(".", "p"),
        split="train",
        purpose=f"Single-joint sinusoidal sweeps for Coulomb/viscous friction around {peak_velocity:.2f} rad/s peak.",
        trajectory=trajectory,
        hold_after_sec=hold_after_sec,
    )


def _make_static_hold_phase(
    *,
    joint_names: list[str],
    sample_rate: float,
    amplitude_scale: float,
    hold_sec: float,
    transition_sec: float,
    max_velocity: float,
    max_acceleration: float,
) -> ExcitationPhase:
    center = np.asarray(FRANKA_CENTER, dtype=np.float64)
    amps = amplitude_scale * np.asarray(FRANKA_AMPLITUDES, dtype=np.float64)
    poses = [
        center,
        center + np.array([0.5, -0.3, 0.3, 0.2, -0.4, 0.2, -0.3]) * amps,
        center + np.array([-0.4, 0.2, -0.5, -0.1, 0.3, -0.3, 0.4]) * amps,
        center + np.array([0.2, 0.4, -0.2, 0.3, 0.4, -0.2, -0.5]) * amps,
        center,
    ]
    times = [0.0]
    positions = [poses[0]]
    t = 0.0
    for pose in poses[1:]:
        start = positions[-1]
        transition_count = max(3, int(round(transition_sec * sample_rate)) + 1)
        for idx in range(1, transition_count):
            u = idx / float(transition_count - 1)
            smooth = 0.5 - 0.5 * math.cos(math.pi * u)
            times.append(t + transition_sec * u)
            positions.append((1.0 - smooth) * start + smooth * pose)
        t += transition_sec
        hold_count = max(2, int(round(hold_sec * sample_rate)) + 1)
        for idx in range(1, hold_count):
            times.append(t + hold_sec * idx / float(hold_count - 1))
            positions.append(pose)
        t += hold_sec

    trajectory = _build_trajectory(
        joint_names,
        np.asarray(times),
        np.vstack(positions),
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )
    return ExcitationPhase(
        "static_holds",
        "train",
        "Static offset poses for gravity, bias, and near-zero velocity residuals.",
        trajectory,
        hold_after_sec=0.0,
    )


def build_excitation_suite(args: argparse.Namespace) -> list[ExcitationPhase]:
    phases: list[ExcitationPhase] = []
    phases.append(
        _make_multisine_phase(
            name="warmup_multisine",
            split="train",
            purpose="Low-amplitude warmup to settle controller, friction, and temperature effects.",
            joint_names=args.joints,
            cycles=args.warmup_cycles,
            sample_rate=args.sample_rate,
            base_period=args.base_period,
            amplitude_scale=args.amplitude_scale * 0.40,
            variant="warmup",
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    for peak_velocity in args.friction_peak_velocity:
        phases.append(
            _make_friction_sweep_phase(
                joint_names=args.joints,
                sample_rate=args.sample_rate,
                amplitude_scale=args.friction_amplitude_scale,
                peak_velocity=peak_velocity,
                repeats=args.single_joint_repeats,
                max_acceleration=args.max_joint_acceleration,
                hold_after_sec=args.hold_after_sec,
            )
        )
    phases.append(
        _make_multisine_phase(
            name="coupled_multisine_train",
            split="train",
            purpose="Coupled joint excitation for inertial, Coriolis, gravity, and actuator tracking parameters.",
            joint_names=args.joints,
            cycles=args.train_cycles,
            sample_rate=args.sample_rate,
            base_period=args.base_period,
            amplitude_scale=args.amplitude_scale,
            variant="train",
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    phases.append(
        _make_multisine_phase(
            name="coupled_multisine_fast",
            split="train",
            purpose="Lower-amplitude faster coupled excitation to emphasize acceleration-dependent terms.",
            joint_names=args.joints,
            cycles=max(1, args.train_cycles // 2),
            sample_rate=args.sample_rate,
            base_period=max(3.0, args.base_period * 0.65),
            amplitude_scale=args.amplitude_scale * 0.65,
            variant="fast",
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    phases.append(
        _make_static_hold_phase(
            joint_names=args.joints,
            sample_rate=args.sample_rate,
            amplitude_scale=args.amplitude_scale,
            hold_sec=args.static_hold_sec,
            transition_sec=args.static_transition_sec,
            max_velocity=args.max_joint_velocity * 0.6,
            max_acceleration=args.max_joint_acceleration * 0.6,
        )
    )
    phases.append(
        _make_multisine_phase(
            name="coupled_multisine_validation",
            split="validation",
            purpose="Held-out excitation with different phases/harmonics; do not tune parameters on this phase.",
            joint_names=args.joints,
            cycles=args.validation_cycles,
            sample_rate=args.sample_rate,
            base_period=args.base_period * 0.85,
            amplitude_scale=args.amplitude_scale * 0.85,
            variant="validation",
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    return phases


def write_manifest(path: Path, phases: list[ExcitationPhase], args: argparse.Namespace) -> None:
    manifest = {
        "schema": "franka_sysid_collection_v2",
        "description": "Methodical joint-space SysID suite with train and held-out validation phases.",
        "notes": [
            "Reference commands are explicit FollowJointTrajectory samples, not MoveIt-retimed sine waypoints.",
            "Use train phases for fitting and validation phases for held-out error checks.",
            "All generated joint positions are clipped to conservative Franka limits.",
        ],
        "sample_rate_hz": args.sample_rate,
        "max_joint_velocity_rad_s": args.max_joint_velocity,
        "max_joint_acceleration_rad_s2": args.max_joint_acceleration,
        "phases": [
            {
                "name": phase.name,
                "split": phase.split,
                "purpose": phase.purpose,
                "duration_sec": phase.duration,
                "points": len(phase.trajectory.points),
                "hold_after_sec": phase.hold_after_sec,
                "use_moveit_start": phase.use_moveit_start,
            }
            for phase in phases
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def append_phase_event(path: Path, node: Node, phase: ExcitationPhase, event: str) -> None:
    now = node.get_clock().now().to_msg()
    payload = {
        "event": event,
        "phase": phase.name,
        "split": phase.split,
        "ros_time_sec": float(now.sec) + 1e-9 * float(now.nanosec),
        "wall_time": time.time(),
        "monotonic_time": time.monotonic(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def _robot_trajectory_from_joint_trajectory(joint_trajectory: JointTrajectory) -> RobotTrajectory:
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory = joint_trajectory
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute on the robot. Without this, dry-run only.")
    parser.add_argument("--group", default="panda_arm")
    parser.add_argument("--joints", nargs="+", default=FRANKA_JOINTS)
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--telemetry-topic", default="/sysid/controller_state")
    parser.add_argument("--follow-action", default="/panda_arm_controller/follow_joint_trajectory")
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--execute-action", default="/execute_trajectory")
    parser.add_argument("--skip-moveit-start", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--storage", default="mcap")
    parser.add_argument("--no-record-bag", action="store_true")
    parser.add_argument("--record-topic", action="append", default=["/joint_states"])
    parser.add_argument("--controllers", nargs="*", default=[])
    parser.add_argument("--planner-id", default="")
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--allowed-planning-time", type=float, default=5.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.01)
    parser.add_argument("--action-server-timeout", type=float, default=10.0)
    parser.add_argument("--planning-timeout", type=float, default=30.0)
    parser.add_argument("--execution-timeout", type=float, default=90.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--base-period", type=float, default=8.0)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--train-cycles", type=int, default=4)
    parser.add_argument("--validation-cycles", type=int, default=2)
    parser.add_argument("--amplitude-scale", type=float, default=0.70)
    parser.add_argument("--friction-amplitude-scale", type=float, default=0.55)
    parser.add_argument("--friction-peak-velocity", nargs="+", type=float, default=[0.12, 0.28])
    parser.add_argument("--single-joint-repeats", type=int, default=1)
    parser.add_argument("--max-joint-velocity", type=float, default=0.65)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.50)
    parser.add_argument("--static-hold-sec", type=float, default=2.0)
    parser.add_argument("--static-transition-sec", type=float, default=4.0)
    parser.add_argument("--hold-after-sec", type=float, default=0.75)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--include-effort", action="store_true")
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main() -> int:
    args = parse_args()
    if list(args.joints) != FRANKA_JOINTS:
        print(
            "franka_sysid_collect_v2 currently expects the full Panda arm joint list in the default order. "
            "Use v1 or update the v2 excitation constants for a custom joint set.",
            file=sys.stderr,
        )
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"franka_sysid_v2_{stamp}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = build_excitation_suite(args)
    topic_map_path = output_dir / "franka_sysid_topic_map.yaml"
    manifest_path = output_dir / "collection_manifest.json"
    phase_events_path = output_dir / "phase_events.jsonl"
    write_topic_map(topic_map_path, args.telemetry_topic, args.joints, args.include_effort)
    write_manifest(manifest_path, phases, args)

    metadata = {
        "schema": "franka_sysid_v2_run",
        "group": args.group,
        "joints": args.joints,
        "telemetry_topic": args.telemetry_topic,
        "joint_states_topic": args.joint_states_topic,
        "follow_action": args.follow_action,
        "move_group_action": args.move_group_action,
        "execute_action": args.execute_action,
        "topic_map": str(topic_map_path),
        "manifest": str(manifest_path),
        "phase_events": str(phase_events_path),
        "design": "direct_joint_trajectory_excitation_with_moveit_start_repositioning",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rclpy.init(args=sys.argv)
    telemetry = SysIdTelemetryPublisher(args.joints, args.joint_states_topic, args.telemetry_topic)
    executor = MultiThreadedExecutor()
    executor.add_node(telemetry)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    logger = telemetry.get_logger()
    logger.info(f"Wrote topic map: {topic_map_path}")
    logger.info(f"Wrote collection manifest: {manifest_path}")
    logger.info(f"Generated {len(phases)} methodical SysID phases")

    if not telemetry.wait_for_joint_state(timeout_sec=10.0):
        logger.error(f"No joint state received on {args.joint_states_topic}")
        executor.shutdown()
        telemetry.destroy_node()
        rclpy.shutdown()
        return 2

    follow_client = FollowTrajectoryClient(telemetry, args.follow_action)
    moveit = None
    if args.execute:
        follow_client.wait_for_server(args.action_server_timeout)
        if not args.skip_moveit_start:
            moveit = MoveItActionClient(
                telemetry,
                move_group_action=args.move_group_action,
                execute_action=args.execute_action,
                group_name=args.group,
                joint_names=args.joints,
                planner_id=args.planner_id,
                pipeline_id=args.pipeline_id,
                planning_attempts=args.planning_attempts,
                allowed_planning_time=args.allowed_planning_time,
                goal_tolerance=args.goal_tolerance,
                velocity_scale=min(args.max_joint_velocity, 0.25),
                acceleration_scale=min(args.max_joint_acceleration, 0.25),
            )
            moveit.wait_for_servers(args.action_server_timeout)

    recorder = None
    if args.execute and not args.no_record_bag:
        topics = unique_ordered([args.telemetry_topic, *args.record_topic])
        recorder = BagRecorder(output_dir / "bag", topics, args.storage)
        logger.info(f"Starting bag recorder for topics: {topics}")
        recorder.start()

    def stop_requested(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_requested)

    try:
        for phase_i, phase in enumerate(phases, start=1):
            logger.info(
                f"Phase {phase_i}/{len(phases)} {phase.name}: {phase.duration:.2f}s, "
                f"{len(phase.trajectory.points)} points, split={phase.split}"
            )
            append_phase_event(phase_events_path, telemetry, phase, "planned")
            if not args.execute:
                continue

            if moveit is not None and phase.use_moveit_start:
                logger.info(f"MoveIt reposition to start of {phase.name}")
                planned = moveit.plan(phase.start_position, telemetry.latest_feedback(), args.planning_timeout)
                duration = duration_to_sec(planned.joint_trajectory.points[-1].time_from_start)
                telemetry.set_trajectory(planned.joint_trajectory)
                time.sleep(args.settle_sec)
                moveit.execute_trajectory(
                    planned,
                    controllers=args.controllers,
                    timeout_sec=max(args.execution_timeout, duration + args.execution_timeout),
                )
                telemetry.hold_last_reference()
                time.sleep(args.hold_after_sec)

            append_phase_event(phase_events_path, telemetry, phase, "start")
            telemetry.set_trajectory(phase.trajectory)
            time.sleep(args.settle_sec)
            follow_client.execute(
                phase.trajectory,
                timeout_sec=max(args.execution_timeout, phase.duration + args.execution_timeout),
            )
            telemetry.hold_last_reference()
            append_phase_event(phase_events_path, telemetry, phase, "end")
            time.sleep(max(0.0, phase.hold_after_sec))

        logger.info("SysID v2 collection complete")
        return 0

    except KeyboardInterrupt:
        logger.warn("Interrupted by user")
        return 130
    finally:
        telemetry.disable()
        if recorder is not None:
            logger.info("Stopping bag recorder")
            recorder.stop()
        executor.shutdown()
        telemetry.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
