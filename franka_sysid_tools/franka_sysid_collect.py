"""Collect Franka free-space System Identification data with MoveIt 2.

The node plans a conservative joint-space excitation, mirrors measured
``/joint_states`` plus the planned reference into one controller-state-like
topic, records a bag, and writes the Isaac SysID topic mapping next to it.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


FRANKA_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_CENTER = [0.0, -0.75, 0.0, -2.20, 0.0, 1.75, 0.80]
FRANKA_AMPLITUDES = [0.35, 0.25, 0.35, 0.22, 0.35, 0.22, 0.35]
FRANKA_LIMITS = [
    (-2.70, 2.70),
    (-1.55, 1.55),
    (-2.70, 2.70),
    (-2.95, -0.25),
    (-2.70, 2.70),
    (0.15, 3.45),
    (-2.70, 2.70),
]


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def duration_to_sec(duration) -> float:
    return float(duration.sec) + 1e-9 * float(duration.nanosec)


def unique_ordered(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def make_point(
    positions: list[float],
    velocities: list[float] | None = None,
    efforts: list[float] | None = None,
) -> JointTrajectoryPoint:
    point = JointTrajectoryPoint()
    point.positions = list(positions)
    point.velocities = list(velocities or [0.0] * len(positions))
    if efforts is not None:
        point.effort = list(efforts)
    return point


@dataclass
class ReferenceSample:
    positions: list[float]
    velocities: list[float]


class TrajectoryReference:
    """Interpolates a planned trajectory into the desired SysID joint order."""

    def __init__(self, trajectory: JointTrajectory, joint_names: list[str]):
        self.joint_names = list(joint_names)
        self.source_joint_names = list(trajectory.joint_names)
        self.points = list(trajectory.points)
        if not self.points:
            raise ValueError("planned trajectory has no points")

        self.source_index = {name: i for i, name in enumerate(self.source_joint_names)}
        missing = [joint for joint in self.joint_names if joint not in self.source_index]
        if missing:
            raise ValueError(f"planned trajectory is missing joints: {missing}")

        self.times = [duration_to_sec(point.time_from_start) for point in self.points]
        if len(self.times) > 1 and self.times[-1] <= 0.0:
            self.times = [float(i) * 0.1 for i in range(len(self.points))]

    @property
    def duration(self) -> float:
        return max(0.0, self.times[-1])

    def _ordered(self, point: JointTrajectoryPoint, field: str) -> list[float]:
        values = list(getattr(point, field))
        if len(values) != len(self.source_joint_names):
            return [0.0] * len(self.joint_names)
        return [float(values[self.source_index[joint]]) for joint in self.joint_names]

    def sample(self, elapsed: float) -> ReferenceSample:
        if elapsed <= self.times[0]:
            return ReferenceSample(
                self._ordered(self.points[0], "positions"),
                self._ordered(self.points[0], "velocities"),
            )

        for i in range(len(self.points) - 1):
            t0 = self.times[i]
            t1 = self.times[i + 1]
            if t0 <= elapsed <= t1:
                p0 = self._ordered(self.points[i], "positions")
                p1 = self._ordered(self.points[i + 1], "positions")
                v0 = self._ordered(self.points[i], "velocities")
                v1 = self._ordered(self.points[i + 1], "velocities")
                u = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
                q = [(1.0 - u) * a + u * b for a, b in zip(p0, p1)]
                dq = [(1.0 - u) * a + u * b for a, b in zip(v0, v1)]
                if not any(abs(value) > 1e-12 for value in dq) and t1 > t0:
                    dq = [(b - a) / (t1 - t0) for a, b in zip(p0, p1)]
                return ReferenceSample(q, dq)

        return ReferenceSample(
            self._ordered(self.points[-1], "positions"),
            self._ordered(self.points[-1], "velocities"),
        )


class SysIdTelemetryPublisher(Node):
    """Publishes measured state and planned reference on one aligned topic."""

    def __init__(self, joint_names: list[str], joint_states_topic: str, telemetry_topic: str):
        super().__init__("franka_sysid_telemetry_publisher")
        self.joint_names = list(joint_names)
        self.joint_states_topic = joint_states_topic
        self.telemetry_topic = telemetry_topic
        self.pub = self.create_publisher(JointTrajectoryControllerState, telemetry_topic, 10)
        self.sub = self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 50)

        self._lock = threading.Lock()
        self._have_state = threading.Event()
        self._enabled = False
        self._reference: TrajectoryReference | None = None
        self._reference_start_monotonic = 0.0
        self._hold_reference: list[float] | None = None
        self._last_feedback: tuple[list[float], list[float], list[float]] | None = None
        self._last_warn_time = 0.0

    def wait_for_joint_state(self, timeout_sec: float) -> bool:
        return self._have_state.wait(timeout=timeout_sec)

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._reference = None

    def set_trajectory(self, trajectory: JointTrajectory) -> float:
        reference = TrajectoryReference(trajectory, self.joint_names)
        with self._lock:
            self._reference = reference
            self._reference_start_monotonic = time.monotonic()
            self._hold_reference = None
            self._enabled = True
        return reference.duration

    def hold_last_reference(self) -> None:
        with self._lock:
            if self._last_feedback is not None:
                self._hold_reference = list(self._last_feedback[0])
            self._reference = None

    def _extract_feedback(self, msg: JointState) -> tuple[list[float], list[float], list[float]] | None:
        index = {name: i for i, name in enumerate(msg.name)}
        missing = [joint for joint in self.joint_names if joint not in index]
        if missing:
            now = time.monotonic()
            if now - self._last_warn_time > 2.0:
                self.get_logger().warn(f"{self.joint_states_topic} missing requested joints: {missing}")
                self._last_warn_time = now
            return None

        def take(values, default: float = 0.0) -> list[float]:
            out = []
            for joint in self.joint_names:
                i = index[joint]
                out.append(float(values[i]) if i < len(values) else default)
            return out

        return take(msg.position), take(msg.velocity), take(msg.effort)

    def _sample_reference(self, feedback_q: list[float]) -> ReferenceSample:
        with self._lock:
            reference = self._reference
            start = self._reference_start_monotonic
            hold = self._hold_reference

        if reference is not None:
            return reference.sample(time.monotonic() - start)
        if hold is not None:
            return ReferenceSample(list(hold), [0.0] * len(hold))
        return ReferenceSample(list(feedback_q), [0.0] * len(feedback_q))

    def _set_state_field(self, msg: JointTrajectoryControllerState, name: str, point: JointTrajectoryPoint) -> None:
        if hasattr(msg, name):
            setattr(msg, name, point)

    def _on_joint_state(self, msg: JointState) -> None:
        feedback = self._extract_feedback(msg)
        if feedback is None:
            return

        q, dq, effort = feedback
        with self._lock:
            self._last_feedback = feedback
            enabled = self._enabled
        self._have_state.set()

        if not enabled:
            return

        reference = self._sample_reference(q)
        err_q = [a - b for a, b in zip(reference.positions, q)]
        err_dq = [a - b for a, b in zip(reference.velocities, dq)]

        out = JointTrajectoryControllerState()
        out.header = msg.header
        if stamp_to_sec(out.header.stamp) <= 0.0:
            out.header.stamp = self.get_clock().now().to_msg()
        out.joint_names = list(self.joint_names)

        reference_point = make_point(reference.positions, reference.velocities)
        feedback_point = make_point(q, dq, effort)
        error_point = make_point(err_q, err_dq)

        self._set_state_field(out, "reference", reference_point)
        self._set_state_field(out, "feedback", feedback_point)
        self._set_state_field(out, "error", error_point)
        self._set_state_field(out, "desired", reference_point)
        self._set_state_field(out, "actual", feedback_point)

        if hasattr(out, "speed_scaling_factor"):
            out.speed_scaling_factor = 1.0

        self.pub.publish(out)


class BagRecorder:
    def __init__(self, output_dir: Path, topics: list[str], storage: str):
        self.output_dir = output_dir
        self.topics = topics
        self.storage = storage
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record", "-s", self.storage, "-o", str(self.output_dir), *self.topics]
        self.proc = subprocess.Popen(cmd)
        time.sleep(1.0)

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=3.0)
        self.proc = None


def generate_excitation_waypoints(cycles: int, samples_per_cycle: int, amplitude_scale: float) -> list[list[float]]:
    waypoints = [list(FRANKA_CENTER)]
    phases = [0.0, 1.7, 0.9, 2.2, 1.1, 2.8, 0.4]
    freqs = [1.0, 1.0, 1.7, 1.3, 2.1, 1.5, 2.4]
    count = max(2, cycles * samples_per_cycle)

    for k in range(count):
        theta = 2.0 * math.pi * float(k + 1) / float(samples_per_cycle)
        q = []
        for center, amp, phase, freq, limits in zip(FRANKA_CENTER, FRANKA_AMPLITUDES, phases, freqs, FRANKA_LIMITS):
            value = center + amplitude_scale * amp * math.sin(freq * theta + phase)
            lo, hi = limits
            q.append(min(max(value, lo), hi))
        waypoints.append(q)

    waypoints.append(list(FRANKA_CENTER))
    return waypoints


def write_topic_map(path: Path, telemetry_topic: str, joint_names: list[str], include_effort: bool) -> None:
    torque_block = (
        f"torque_topic: {telemetry_topic}\n"
        "torque_fields:\n"
        "  - feedback.effort\n"
        "  - actual.effort\n"
        if include_effort
        else "torque_topic: null\ntorque_fields:\n  - feedback.effort\n  - actual.effort\n"
    )
    joints_yaml = "\n".join(f"  - {name}" for name in joint_names)
    text = f"""time_topic: null
time_field: header.stamp

position_topic: {telemetry_topic}
position_fields:
  - feedback.positions
  - actual.positions

velocity_topic: {telemetry_topic}
velocity_fields:
  - feedback.velocities
  - actual.velocities

command_topic: {telemetry_topic}
command_fields:
  - reference.positions
  - desired.positions

{torque_block}
end_effector_pose_topic: null
end_effector_pose_fields:
  - pose

contact_force_topic: null
contact_force_fields:
  - wrench.force

joint_names:
{joints_yaml}
"""
    path.write_text(text, encoding="utf-8")


def plan_to_joint_goal(moveit: MoveItPy, planning_component, group_name: str, q_goal: list[float]):
    robot_model = moveit.get_robot_model()
    goal_state = RobotState(robot_model)
    goal_state.set_joint_group_positions(group_name, q_goal)
    if hasattr(goal_state, "update"):
        goal_state.update()

    planning_component.set_start_state_to_current_state()
    planning_component.set_goal_state(robot_state=goal_state)
    return planning_component.plan()


def retime_trajectory(trajectory, velocity_scale: float, acceleration_scale: float, logger) -> None:
    if hasattr(trajectory, "apply_ruckig_smoothing"):
        if trajectory.apply_ruckig_smoothing(velocity_scale, acceleration_scale):
            return
        logger.warn("Ruckig smoothing failed; trying TOTG if available")

    if hasattr(trajectory, "apply_totg_time_parameterization"):
        if not trajectory.apply_totg_time_parameterization(velocity_scale, acceleration_scale):
            logger.warn("TOTG time parameterization failed; using planner timing")


def extract_joint_trajectory(trajectory) -> JointTrajectory:
    if isinstance(trajectory, JointTrajectory):
        return trajectory

    if hasattr(trajectory, "joint_trajectory"):
        joint_traj = trajectory.joint_trajectory
        if joint_traj.points:
            return joint_traj

    if hasattr(trajectory, "get_robot_trajectory_msg"):
        try:
            robot_traj_msg = trajectory.get_robot_trajectory_msg()
        except TypeError:
            robot_traj_msg = trajectory.get_robot_trajectory_msg(None)
        joint_traj = robot_traj_msg.joint_trajectory
        if joint_traj.points:
            return joint_traj

    raise RuntimeError("Could not extract a trajectory_msgs/JointTrajectory from the MoveIt plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute on the robot. Without this, plan only.")
    parser.add_argument("--group", default="panda_arm")
    parser.add_argument("--joints", nargs="+", default=FRANKA_JOINTS)
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--telemetry-topic", default="/sysid/controller_state")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--storage", default="mcap")
    parser.add_argument("--no-record-bag", action="store_true")
    parser.add_argument("--record-topic", action="append", default=["/joint_states"])
    parser.add_argument("--controllers", nargs="*", default=[])
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--samples-per-cycle", type=int, default=6)
    parser.add_argument("--amplitude-scale", type=float, default=0.75)
    parser.add_argument("--velocity-scale", type=float, default=0.25)
    parser.add_argument("--acceleration-scale", type=float, default=0.25)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--segment-pause-sec", type=float, default=0.25)
    parser.add_argument("--include-effort", action="store_true")
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"franka_sysid_{stamp}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_map_path = output_dir / "franka_sysid_topic_map.yaml"
    write_topic_map(topic_map_path, args.telemetry_topic, args.joints, args.include_effort)

    metadata = {
        "group": args.group,
        "joints": args.joints,
        "telemetry_topic": args.telemetry_topic,
        "joint_states_topic": args.joint_states_topic,
        "cycles": args.cycles,
        "samples_per_cycle": args.samples_per_cycle,
        "amplitude_scale": args.amplitude_scale,
        "velocity_scale": args.velocity_scale,
        "acceleration_scale": args.acceleration_scale,
        "topic_map": str(topic_map_path),
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

    if not telemetry.wait_for_joint_state(timeout_sec=10.0):
        logger.error(f"No joint state received on {args.joint_states_topic}")
        executor.shutdown()
        telemetry.destroy_node()
        rclpy.shutdown()
        return 2

    moveit = MoveItPy(node_name="franka_sysid_moveit")
    arm = moveit.get_planning_component(args.group)

    waypoints = generate_excitation_waypoints(args.cycles, args.samples_per_cycle, args.amplitude_scale)
    logger.info(f"Generated {len(waypoints)} joint-space waypoints")

    recorder = None
    if args.execute and not args.no_record_bag:
        topics = unique_ordered([args.telemetry_topic, *args.record_topic])
        recorder = BagRecorder(output_dir / "bag", topics, args.storage)
        logger.info(f"Starting bag recorder for topics: {topics}")
        recorder.start()

    try:
        for i, q_goal in enumerate(waypoints, start=1):
            logger.info(f"Planning segment {i}/{len(waypoints)}")
            plan_result = plan_to_joint_goal(moveit, arm, args.group, q_goal)
            if not plan_result:
                raise RuntimeError(f"Planning failed for waypoint {i}: {q_goal}")

            trajectory = plan_result.trajectory
            retime_trajectory(trajectory, args.velocity_scale, args.acceleration_scale, logger)
            joint_traj = extract_joint_trajectory(trajectory)
            duration = duration_to_sec(joint_traj.points[-1].time_from_start)
            logger.info(f"Segment {i} duration: {duration:.3f} s")

            if not args.execute:
                continue

            telemetry.set_trajectory(joint_traj)
            time.sleep(args.settle_sec)
            moveit.execute(trajectory, controllers=args.controllers)
            telemetry.hold_last_reference()
            time.sleep(args.segment_pause_sec)

        logger.info("SysID collection complete")
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
