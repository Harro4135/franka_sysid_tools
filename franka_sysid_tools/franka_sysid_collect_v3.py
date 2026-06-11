"""Franka SysID collection with physical-regressor D-optimal NLP design.

V3 keeps the v2 safety and execution path: MoveIt can reposition to each phase
start, direct ``FollowJointTrajectory`` references are recorded as the designed
commands, and optional MoveIt state-validity checks run before execution.

The difference is the coupled excitation design. V3 uses a finite Fourier
series parameterization and solves a constrained nonlinear program with CasADi
/ IPOPT. The D-optimal objective is evaluated on the identifiable base columns
of Pinocchio's physical joint-torque regressor, extracted from the robot URDF.
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
from moveit_msgs.msg import RobotState, RobotTrajectory
from moveit_msgs.srv import GetStateValidity
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
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
from .dynamics import BaseRegressorModel, load_base_regressor_model


TRAIN_PHASES = {"warmup_multisine", "d_optimal_train", "d_optimal_fast", "static_holds"}
VALIDATION_PHASES = {"d_optimal_validation"}


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


class MoveItStateValidityChecker:
    """Waypoint collision checker using MoveIt's GetStateValidity service."""

    def __init__(self, node: Node, service_name: str, group_name: str, joint_names: list[str]):
        self.node = node
        self.service_name = service_name
        self.group_name = group_name
        self.joint_names = list(joint_names)
        self.client = node.create_client(GetStateValidity, service_name)

    def wait_for_service(self, timeout_sec: float) -> None:
        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"MoveIt state-validity service is not available: {self.service_name}. "
                "Launch move_group with a planning scene monitor, or pass --no-collision-check."
            )

    def _wait_future(self, future, timeout_sec: float, description: str):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {description}")
            time.sleep(0.01)
        return future.result()

    def _request_for_positions(self, positions: list[float]) -> GetStateValidity.Request:
        if len(positions) != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} joint positions, got {len(positions)}")
        joint_state = JointState()
        joint_state.name = list(self.joint_names)
        joint_state.position = [float(value) for value in positions]

        robot_state = RobotState()
        robot_state.joint_state = joint_state
        robot_state.is_diff = True

        request = GetStateValidity.Request()
        request.robot_state = robot_state
        request.group_name = self.group_name
        return request

    @staticmethod
    def _format_contacts(response) -> str:
        contacts = list(getattr(response, "contacts", []))
        if not contacts:
            return "no contact details returned"
        pieces = []
        for contact in contacts[:5]:
            body_a = getattr(contact, "body_name_1", "?")
            body_b = getattr(contact, "body_name_2", "?")
            depth = getattr(contact, "depth", 0.0)
            pieces.append(f"{body_a} <-> {body_b} depth={float(depth):.4g}")
        suffix = "" if len(contacts) <= 5 else f", +{len(contacts) - 5} more"
        return "; ".join(pieces) + suffix

    def check_positions(self, positions: list[float], timeout_sec: float) -> tuple[bool, str]:
        future = self.client.call_async(self._request_for_positions(positions))
        response = self._wait_future(future, timeout_sec, "state-validity response")
        if response is None:
            raise RuntimeError("MoveIt state-validity service returned no response")
        if bool(getattr(response, "valid", False)):
            return True, ""
        return False, self._format_contacts(response)

    def check_trajectory(
        self,
        trajectory: JointTrajectory,
        *,
        phase_name: str,
        stride: int,
        timeout_sec: float,
    ) -> int:
        if list(trajectory.joint_names) != self.joint_names:
            raise ValueError(
                f"{phase_name}: trajectory joints {list(trajectory.joint_names)} do not match "
                f"collision checker joints {self.joint_names}"
            )
        if not trajectory.points:
            raise ValueError(f"{phase_name}: trajectory has no points to collision-check")

        point_count = len(trajectory.points)
        step = max(1, int(stride))
        sample_indices = set(range(0, point_count, step))
        sample_indices.add(point_count - 1)

        checked = 0
        for point_i in sorted(sample_indices):
            point = trajectory.points[point_i]
            valid, detail = self.check_positions(list(point.positions), timeout_sec)
            checked += 1
            if not valid:
                t = duration_to_sec(point.time_from_start)
                raise RuntimeError(
                    f"{phase_name}: MoveIt collision check failed at point {point_i}/{point_count - 1} "
                    f"(t={t:.3f}s): {detail}"
                )
        return checked


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


def _build_trajectory_with_derivatives(
    joint_names: list[str],
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
) -> JointTrajectory:
    times = np.asarray(times, dtype=np.float64)
    positions = _clamp_positions(np.asarray(positions, dtype=np.float64))
    velocities = np.asarray(velocities, dtype=np.float64)
    accelerations = np.asarray(accelerations, dtype=np.float64)
    if (
        times.ndim != 1
        or positions.ndim != 2
        or velocities.shape != positions.shape
        or accelerations.shape != positions.shape
        or len(times) != positions.shape[0]
    ):
        raise ValueError("times must be (T,) and q/dq/ddq must be matching (T, N) arrays")
    if len(times) < 2:
        raise ValueError("trajectory must contain at least two points")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")

    trajectory = JointTrajectory()
    trajectory.joint_names = list(joint_names)
    for t, q, dq, ddq in zip(times, positions, velocities, accelerations):
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
    trajectory = _build_trajectory_with_derivatives(
        joint_names,
        times,
        positions,
        velocities,
        accelerations,
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


def _fourier_unroll_numpy(
    x: np.ndarray,
    times: np.ndarray,
    *,
    joint_count: int,
    harmonic_count: int,
    base_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    q0_count = joint_count
    coeff_count = joint_count * harmonic_count
    q0 = x[:q0_count]
    a = x[q0_count : q0_count + coeff_count].reshape(joint_count, harmonic_count)
    b = x[q0_count + coeff_count : q0_count + 2 * coeff_count].reshape(joint_count, harmonic_count)
    omega = 2.0 * math.pi / float(base_period)
    q = np.tile(q0, (len(times), 1))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    for harmonic_i in range(harmonic_count):
        harmonic = float(harmonic_i + 1)
        wt = omega * harmonic * times
        sin_wt = np.sin(wt)[:, np.newaxis]
        cos_wt = np.cos(wt)[:, np.newaxis]
        denom = omega * harmonic
        q += (a[:, harmonic_i] / denom) * sin_wt - (b[:, harmonic_i] / denom) * cos_wt
        dq += a[:, harmonic_i] * cos_wt + b[:, harmonic_i] * sin_wt
        ddq += -(a[:, harmonic_i] * denom) * sin_wt + (b[:, harmonic_i] * denom) * cos_wt
    return q, dq, ddq


class _PhysicalLogDetCallback:
    """CasADi callback wrapper around Pinocchio's numeric base regressor."""

    def __new__(
        cls,
        *,
        name: str,
        regressor_model: BaseRegressorModel,
        times: np.ndarray,
        joint_count: int,
        harmonic_count: int,
        base_period: float,
        ridge: float,
        condition_penalty: float,
    ):
        try:
            import casadi as ca
        except ImportError as exc:
            raise RuntimeError(
                "franka_sysid_collect_v3 requires CasADi for NLP trajectory optimization. "
                "Install python3-casadi or pip package casadi in the ROS environment."
            ) from exc

        class Callback(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.regressor_model = regressor_model
                self.times = np.asarray(times, dtype=np.float64)
                self.joint_count = int(joint_count)
                self.harmonic_count = int(harmonic_count)
                self.base_period = float(base_period)
                self.ridge = float(ridge)
                self.condition_penalty = float(condition_penalty)
                variable_count = self.joint_count * (1 + 2 * self.harmonic_count)
                self._input_sparsity = ca.Sparsity.dense(variable_count, 1)
                self.construct(name, {"enable_fd": True})

            def get_n_in(self):
                return 1

            def get_n_out(self):
                return 1

            def get_sparsity_in(self, _index):
                return self._input_sparsity

            def get_sparsity_out(self, _index):
                return ca.Sparsity.dense(1, 1)

            def eval(self, arg):
                x = np.asarray(arg[0], dtype=np.float64).reshape(-1)
                q, dq, ddq = _fourier_unroll_numpy(
                    x,
                    self.times,
                    joint_count=self.joint_count,
                    harmonic_count=self.harmonic_count,
                    base_period=self.base_period,
                )
                score = self.regressor_model.logdet_information(
                    q,
                    dq,
                    ddq,
                    ridge=self.ridge,
                    condition_penalty=self.condition_penalty,
                )
                return [np.asarray([[score]], dtype=np.float64)]

        return Callback()


def _fourier_unroll_symbolic(opti, times: np.ndarray, q0, a, b, base_period: float):
    try:
        import casadi as ca
    except ImportError as exc:
        raise RuntimeError(
            "franka_sysid_collect_v3 requires CasADi for NLP trajectory optimization."
        ) from exc

    omega = 2.0 * math.pi / float(base_period)
    q_rows = []
    dq_rows = []
    ddq_rows = []
    harmonic_count = int(a.shape[1])
    for time_value in times:
        q_t = q0
        dq_t = 0.0 * q0
        ddq_t = 0.0 * q0
        for harmonic_i in range(harmonic_count):
            harmonic = float(harmonic_i + 1)
            wt = omega * harmonic * float(time_value)
            denom = omega * harmonic
            a_col = a[:, harmonic_i]
            b_col = b[:, harmonic_i]
            q_t = q_t + (a_col / denom) * math.sin(wt) - (b_col / denom) * math.cos(wt)
            dq_t = dq_t + a_col * math.cos(wt) + b_col * math.sin(wt)
            ddq_t = ddq_t - (a_col * denom) * math.sin(wt) + (b_col * denom) * math.cos(wt)
        q_rows.append(q_t.T)
        dq_rows.append(dq_t.T)
        ddq_rows.append(ddq_t.T)
    return ca.vertcat(*q_rows), ca.vertcat(*dq_rows), ca.vertcat(*ddq_rows)


def _solve_d_optimal_phase(
    *,
    name: str,
    split: str,
    purpose: str,
    joint_names: list[str],
    regressor_model: BaseRegressorModel,
    cycles: int,
    sample_rate: float,
    base_period: float,
    amplitude_scale: float,
    seed: int,
    harmonic_count: int,
    score_stride: int,
    ridge: float,
    condition_penalty: float,
    ipopt_max_iter: int,
    ipopt_print_level: int,
    ipopt_tolerance: float,
    max_velocity: float,
    max_acceleration: float,
    hold_after_sec: float,
) -> ExcitationPhase:
    try:
        import casadi as ca
    except ImportError as exc:
        raise RuntimeError(
            "franka_sysid_collect_v3 requires CasADi for NLP trajectory optimization. "
            "Install python3-casadi or pip package casadi in the ROS environment."
        ) from exc

    duration = max(base_period, float(cycles) * base_period)
    sample_count = max(3, int(round(duration * sample_rate)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    score_indices = np.arange(0, len(times), max(1, int(score_stride)), dtype=np.int64)
    if score_indices[-1] != len(times) - 1:
        score_indices = np.append(score_indices, len(times) - 1)
    score_times = times[score_indices]

    joint_count = len(joint_names)
    opti = ca.Opti()
    q0 = opti.variable(joint_count)
    a = opti.variable(joint_count, harmonic_count)
    b = opti.variable(joint_count, harmonic_count)
    x = ca.vertcat(q0, ca.reshape(a, -1, 1), ca.reshape(b, -1, 1))

    q_grid, dq_grid, ddq_grid = _fourier_unroll_symbolic(opti, times, q0, a, b, base_period)
    min_q = np.asarray([limit[0] for limit in FRANKA_LIMITS], dtype=np.float64)
    max_q = np.asarray([limit[1] for limit in FRANKA_LIMITS], dtype=np.float64)
    for joint_i in range(joint_count):
        opti.subject_to(opti.bounded(min_q[joint_i], q_grid[:, joint_i], max_q[joint_i]))
        opti.subject_to(opti.bounded(-max_velocity, dq_grid[:, joint_i], max_velocity))
        opti.subject_to(opti.bounded(-max_acceleration, ddq_grid[:, joint_i], max_acceleration))

    opti.subject_to(dq_grid[0, :] == 0.0)
    opti.subject_to(dq_grid[-1, :] == 0.0)
    opti.subject_to(ddq_grid[0, :] == 0.0)
    opti.subject_to(ddq_grid[-1, :] == 0.0)
    opti.subject_to(q_grid[0, :] == q_grid[-1, :])

    callback = _PhysicalLogDetCallback(
        name=f"{name}_physical_logdet",
        regressor_model=regressor_model,
        times=score_times,
        joint_count=joint_count,
        harmonic_count=harmonic_count,
        base_period=base_period,
        ridge=ridge,
        condition_penalty=condition_penalty,
    )
    opti.minimize(-callback(x))

    rng = np.random.default_rng(int(seed))
    center = np.asarray(FRANKA_CENTER, dtype=np.float64)
    opti.set_initial(q0, center)
    coeff_scale = max(0.01, float(amplitude_scale)) * np.asarray(FRANKA_AMPLITUDES, dtype=np.float64)[:, np.newaxis]
    initial_a = rng.normal(0.0, 0.05, size=(joint_count, harmonic_count)) * coeff_scale
    initial_b = rng.normal(0.0, 0.05, size=(joint_count, harmonic_count)) * coeff_scale
    initial_a -= np.mean(initial_a, axis=1, keepdims=True)
    weighted = np.arange(1, harmonic_count + 1, dtype=np.float64)[np.newaxis, :]
    initial_b -= np.sum(weighted * initial_b, axis=1, keepdims=True) / np.sum(weighted)
    opti.set_initial(a, initial_a)
    opti.set_initial(b, initial_b)

    opts = {
        "ipopt.max_iter": int(ipopt_max_iter),
        "ipopt.print_level": int(ipopt_print_level),
        "ipopt.tol": float(ipopt_tolerance),
        "print_time": False,
    }
    opti.solver("ipopt", opts)
    solution = opti.solve()
    x_value = np.asarray(solution.value(x), dtype=np.float64).reshape(-1)
    positions, velocities, accelerations = _fourier_unroll_numpy(
        x_value,
        times,
        joint_count=joint_count,
        harmonic_count=harmonic_count,
        base_period=base_period,
    )
    if np.any(positions < min_q - 1e-6) or np.any(positions > max_q + 1e-6):
        raise RuntimeError(f"{name}: optimized Fourier trajectory violates joint position bounds")
    if np.any(np.abs(velocities) > max_velocity + 1e-6):
        raise RuntimeError(f"{name}: optimized Fourier trajectory violates joint velocity bounds")
    if np.any(np.abs(accelerations) > max_acceleration + 1e-6):
        raise RuntimeError(f"{name}: optimized Fourier trajectory violates joint acceleration bounds")
    score_q, score_dq, score_ddq = _fourier_unroll_numpy(
        x_value,
        score_times,
        joint_count=joint_count,
        harmonic_count=harmonic_count,
        base_period=base_period,
    )
    score = regressor_model.logdet_information(
        score_q,
        score_dq,
        score_ddq,
        ridge=ridge,
        condition_penalty=condition_penalty,
    )

    trajectory = _build_trajectory(
        joint_names,
        times,
        positions,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )
    phase = ExcitationPhase(name, split, purpose, trajectory, hold_after_sec=hold_after_sec)
    phase.d_optimal_score = score
    phase.d_optimal_rank = regressor_model.structural_rank
    phase.d_optimal_full_parameter_count = regressor_model.full_parameter_count
    phase.d_optimal_seed = int(seed)
    phase.d_optimal_harmonics = int(harmonic_count)
    return phase


def build_excitation_suite(args: argparse.Namespace) -> list[ExcitationPhase]:
    regressor_model = load_base_regressor_model(
        urdf_path=args.urdf_path,
        joint_names=args.joints,
        center=FRANKA_CENTER,
        amplitudes=(args.amplitude_scale * np.asarray(FRANKA_AMPLITUDES, dtype=np.float64)).tolist(),
        base_period=args.base_period,
        structural_samples=args.base_regressor_samples,
        rank_tolerance=args.base_regressor_rank_tolerance,
    )
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
    phases.append(
        _solve_d_optimal_phase(
            name="d_optimal_train",
            split="train",
            purpose="D-optimal coupled joint excitation for dense inertial, gravity, velocity, and acceleration information.",
            joint_names=args.joints,
            regressor_model=regressor_model,
            cycles=args.train_cycles,
            sample_rate=args.sample_rate,
            base_period=args.base_period,
            amplitude_scale=args.amplitude_scale,
            seed=args.d_opt_seed,
            harmonic_count=args.fourier_harmonics,
            score_stride=args.d_opt_score_stride,
            ridge=args.d_opt_ridge,
            condition_penalty=args.d_opt_condition_penalty,
            ipopt_max_iter=args.ipopt_max_iter,
            ipopt_print_level=args.ipopt_print_level,
            ipopt_tolerance=args.ipopt_tolerance,
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    phases.append(
        _solve_d_optimal_phase(
            name="d_optimal_fast",
            split="train",
            purpose="Higher-frequency D-optimal excitation emphasizing acceleration-dependent terms.",
            joint_names=args.joints,
            regressor_model=regressor_model,
            cycles=max(1, args.train_cycles // 2),
            sample_rate=args.sample_rate,
            base_period=max(3.0, args.base_period * 0.65),
            amplitude_scale=args.amplitude_scale * 0.65,
            seed=args.d_opt_seed + 101,
            harmonic_count=args.fourier_harmonics,
            score_stride=args.d_opt_score_stride,
            ridge=args.d_opt_ridge,
            condition_penalty=args.d_opt_condition_penalty,
            ipopt_max_iter=args.ipopt_max_iter,
            ipopt_print_level=args.ipopt_print_level,
            ipopt_tolerance=args.ipopt_tolerance,
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
        _solve_d_optimal_phase(
            name="d_optimal_validation",
            split="validation",
            purpose="Held-out D-optimal excitation from a separate initialization seed; do not tune parameters on this phase.",
            joint_names=args.joints,
            regressor_model=regressor_model,
            cycles=args.validation_cycles,
            sample_rate=args.sample_rate,
            base_period=args.base_period * 0.85,
            amplitude_scale=args.amplitude_scale * 0.85,
            seed=args.d_opt_seed + 202,
            harmonic_count=args.fourier_harmonics,
            score_stride=args.d_opt_score_stride,
            ridge=args.d_opt_ridge,
            condition_penalty=args.d_opt_condition_penalty,
            ipopt_max_iter=args.ipopt_max_iter,
            ipopt_print_level=args.ipopt_print_level,
            ipopt_tolerance=args.ipopt_tolerance,
            max_velocity=args.max_joint_velocity,
            max_acceleration=args.max_joint_acceleration,
            hold_after_sec=args.hold_after_sec,
        )
    )
    return phases


def write_manifest(path: Path, phases: list[ExcitationPhase], args: argparse.Namespace) -> None:
    manifest = {
        "schema": "franka_sysid_collection_v3",
        "description": "D-optimal joint-space SysID suite with train and held-out validation phases.",
        "notes": [
            "Reference commands are explicit FollowJointTrajectory samples, not MoveIt-retimed sine waypoints.",
            "Use train phases for fitting and validation phases for held-out error checks.",
            "Coupled excitation phases are finite-Fourier-series NLP solutions with hard sampled q/dq/ddq bounds.",
            "D-optimality is computed against identifiable base columns of Pinocchio's physical torque regressor.",
            "When execution collision checking is enabled, direct trajectory waypoints are sampled through MoveIt GetStateValidity before motion starts.",
        ],
        "sample_rate_hz": args.sample_rate,
        "max_joint_velocity_rad_s": args.max_joint_velocity,
        "max_joint_acceleration_rad_s2": args.max_joint_acceleration,
        "urdf_path": args.urdf_path,
        "base_regressor": {
            "structural_samples": args.base_regressor_samples,
            "rank_tolerance": args.base_regressor_rank_tolerance,
        },
        "d_optimality": {
            "seed": args.d_opt_seed,
            "fourier_harmonics": args.fourier_harmonics,
            "score_stride": args.d_opt_score_stride,
            "ridge": args.d_opt_ridge,
            "condition_penalty": args.d_opt_condition_penalty,
            "ipopt_max_iter": args.ipopt_max_iter,
            "ipopt_tolerance": args.ipopt_tolerance,
        },
        "collision_check": {
            "requested": not args.no_collision_check,
            "service": args.collision_check_service,
            "stride_points": args.collision_check_stride,
            "timeout_sec": args.collision_check_timeout,
        },
        "phases": [
            {
                "name": phase.name,
                "split": phase.split,
                "purpose": phase.purpose,
                "duration_sec": phase.duration,
                "points": len(phase.trajectory.points),
                "hold_after_sec": phase.hold_after_sec,
                "use_moveit_start": phase.use_moveit_start,
                "d_optimal_score": getattr(phase, "d_optimal_score", None),
                "d_optimal_seed": getattr(phase, "d_optimal_seed", None),
                "d_optimal_rank": getattr(phase, "d_optimal_rank", None),
                "d_optimal_full_parameter_count": getattr(phase, "d_optimal_full_parameter_count", None),
                "d_optimal_harmonics": getattr(phase, "d_optimal_harmonics", None),
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
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--collision-check-service", default="/check_state_validity")
    parser.add_argument("--collision-check-stride", type=int, default=10)
    parser.add_argument("--collision-check-timeout", type=float, default=5.0)
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
    parser.add_argument("--urdf-path", default="", help="Fixed-base Panda URDF used by Pinocchio for torque regressors.")
    parser.add_argument("--base-regressor-samples", type=int, default=240)
    parser.add_argument("--base-regressor-rank-tolerance", type=float, default=1e-8)
    parser.add_argument("--fourier-harmonics", type=int, default=5)
    parser.add_argument("--d-opt-seed", type=int, default=20260611)
    parser.add_argument("--d-opt-score-stride", type=int, default=4)
    parser.add_argument("--d-opt-ridge", type=float, default=1e-3)
    parser.add_argument("--d-opt-condition-penalty", type=float, default=0.0)
    parser.add_argument("--ipopt-max-iter", type=int, default=500)
    parser.add_argument("--ipopt-print-level", type=int, default=5)
    parser.add_argument("--ipopt-tolerance", type=float, default=1e-6)
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
            "franka_sysid_collect_v3 currently expects the full Panda arm joint list in the default order. "
            "Use v1 or update the v3 excitation constants for a custom joint set.",
            file=sys.stderr,
        )
        return 2
    if not args.urdf_path:
        print("--urdf-path is required for v3 physical-regressor D-optimal trajectory generation.", file=sys.stderr)
        return 2
    if args.fourier_harmonics < 1:
        print("--fourier-harmonics must be >= 1.", file=sys.stderr)
        return 2
    if args.base_regressor_samples < 2:
        print("--base-regressor-samples must be >= 2.", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"franka_sysid_v3_{stamp}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = build_excitation_suite(args)
    topic_map_path = output_dir / "franka_sysid_topic_map.yaml"
    manifest_path = output_dir / "collection_manifest.json"
    phase_events_path = output_dir / "phase_events.jsonl"
    write_topic_map(topic_map_path, args.telemetry_topic, args.joints, args.include_effort)
    write_manifest(manifest_path, phases, args)

    metadata = {
        "schema": "franka_sysid_v3_run",
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
        "design": "physical_base_regressor_d_optimal_fourier_nlp_with_moveit_start_repositioning",
        "urdf_path": args.urdf_path,
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
    logger.info(f"Generated {len(phases)} D-optimal SysID phases")

    if not telemetry.wait_for_joint_state(timeout_sec=10.0):
        logger.error(f"No joint state received on {args.joint_states_topic}")
        executor.shutdown()
        telemetry.destroy_node()
        rclpy.shutdown()
        return 2

    follow_client = FollowTrajectoryClient(telemetry, args.follow_action)
    moveit = None
    collision_checker = None
    if args.execute:
        follow_client.wait_for_server(args.action_server_timeout)
        if not args.no_collision_check:
            collision_checker = MoveItStateValidityChecker(
                telemetry,
                args.collision_check_service,
                args.group,
                args.joints,
            )
            collision_checker.wait_for_service(args.action_server_timeout)
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

    if collision_checker is not None:
        logger.info(
            f"Preflight collision checking direct phase trajectories via {args.collision_check_service} "
            f"(stride={args.collision_check_stride})"
        )
        total_checked = 0
        for phase in phases:
            checked = collision_checker.check_trajectory(
                phase.trajectory,
                phase_name=phase.name,
                stride=args.collision_check_stride,
                timeout_sec=args.collision_check_timeout,
            )
            total_checked += checked
            logger.info(f"Collision check passed for {phase.name}: {checked} sampled states")
        logger.info(f"Collision preflight passed for all direct phases: {total_checked} sampled states")

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

        logger.info("SysID v3 collection complete")
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
