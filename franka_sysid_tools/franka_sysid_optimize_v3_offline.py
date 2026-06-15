"""Offline physical-regressor D-optimal trajectory generation for Panda SysID.

This script has no ROS 2 imports. It solves the v3 Fourier/NLP trajectory on a
plain workstation with CasADi and Pinocchio, then writes sampled q/dq/ddq data
and a simple inverse-dynamics preview for later ROS/MoveIt simulation or
hardware collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .dynamics import BaseRegressorModel, load_base_regressor_model


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


def _import_casadi():
    try:
        import casadi as ca
    except ImportError as exc:
        raise RuntimeError("Install CasADi before running offline optimization: pip install casadi") from exc
    return ca


def _fourier_unroll_numpy(
    x: np.ndarray,
    times: np.ndarray,
    *,
    joint_count: int,
    harmonic_count: int,
    base_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    coeff_count = joint_count * harmonic_count
    q0 = x[:joint_count]
    a = x[joint_count : joint_count + coeff_count].reshape(joint_count, harmonic_count)
    b = x[joint_count + coeff_count : joint_count + 2 * coeff_count].reshape(joint_count, harmonic_count)
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


def _fourier_unroll_symbolic(times: np.ndarray, q0, a, b, base_period: float):
    ca = _import_casadi()
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


def _physical_logdet_callback(
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
    ca = _import_casadi()

    class Callback(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            variable_count = joint_count * (1 + 2 * harmonic_count)
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
                times,
                joint_count=joint_count,
                harmonic_count=harmonic_count,
                base_period=base_period,
            )
            score = regressor_model.logdet_information(
                q,
                dq,
                ddq,
                ridge=ridge,
                condition_penalty=condition_penalty,
            )
            return [np.asarray([[score]], dtype=np.float64)]

    return Callback()


def solve_offline(args: argparse.Namespace, regressor_model: BaseRegressorModel):
    ca = _import_casadi()
    duration = float(args.cycles) * float(args.base_period)
    sample_count = max(3, int(round(duration * args.sample_rate)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    score_indices = np.arange(0, len(times), max(1, int(args.score_stride)), dtype=np.int64)
    if score_indices[-1] != len(times) - 1:
        score_indices = np.append(score_indices, len(times) - 1)

    joint_count = len(FRANKA_JOINTS)
    opti = ca.Opti()
    q0 = opti.variable(joint_count)
    a = opti.variable(joint_count, args.fourier_harmonics)
    b = opti.variable(joint_count, args.fourier_harmonics)
    x = ca.vertcat(q0, ca.reshape(a, -1, 1), ca.reshape(b, -1, 1))

    q_grid, dq_grid, ddq_grid = _fourier_unroll_symbolic(times, q0, a, b, args.base_period)
    min_q = np.asarray([limit[0] for limit in FRANKA_LIMITS], dtype=np.float64)
    max_q = np.asarray([limit[1] for limit in FRANKA_LIMITS], dtype=np.float64)
    for joint_i in range(joint_count):
        opti.subject_to(opti.bounded(min_q[joint_i], q_grid[:, joint_i], max_q[joint_i]))
        opti.subject_to(opti.bounded(-args.max_joint_velocity, dq_grid[:, joint_i], args.max_joint_velocity))
        opti.subject_to(opti.bounded(-args.max_joint_acceleration, ddq_grid[:, joint_i], args.max_joint_acceleration))
    opti.subject_to(dq_grid[0, :] == 0.0)
    opti.subject_to(dq_grid[-1, :] == 0.0)
    opti.subject_to(ddq_grid[0, :] == 0.0)
    opti.subject_to(ddq_grid[-1, :] == 0.0)
    opti.subject_to(q_grid[0, :] == q_grid[-1, :])

    callback = _physical_logdet_callback(
        name="offline_physical_logdet",
        regressor_model=regressor_model,
        times=times[score_indices],
        joint_count=joint_count,
        harmonic_count=args.fourier_harmonics,
        base_period=args.base_period,
        ridge=args.ridge,
        condition_penalty=args.condition_penalty,
    )
    opti.minimize(-callback(x))

    rng = np.random.default_rng(int(args.seed))
    opti.set_initial(q0, np.asarray(FRANKA_CENTER, dtype=np.float64))
    coeff_scale = max(0.01, float(args.amplitude_scale)) * np.asarray(FRANKA_AMPLITUDES)[:, np.newaxis]
    initial_a = rng.normal(0.0, 0.05, size=(joint_count, args.fourier_harmonics)) * coeff_scale
    initial_b = rng.normal(0.0, 0.05, size=(joint_count, args.fourier_harmonics)) * coeff_scale
    initial_a -= np.mean(initial_a, axis=1, keepdims=True)
    weighted = np.arange(1, args.fourier_harmonics + 1, dtype=np.float64)[np.newaxis, :]
    initial_b -= np.sum(weighted * initial_b, axis=1, keepdims=True) / np.sum(weighted)
    opti.set_initial(a, initial_a)
    opti.set_initial(b, initial_b)

    opti.solver(
        "ipopt",
        {
            "ipopt.max_iter": int(args.ipopt_max_iter),
            "ipopt.print_level": int(args.ipopt_print_level),
            "ipopt.tol": float(args.ipopt_tolerance),
            "print_time": False,
        },
    )
    solution = opti.solve()
    x_value = np.asarray(solution.value(x), dtype=np.float64).reshape(-1)
    q, dq, ddq = _fourier_unroll_numpy(
        x_value,
        times,
        joint_count=joint_count,
        harmonic_count=args.fourier_harmonics,
        base_period=args.base_period,
    )
    score_q, score_dq, score_ddq = _fourier_unroll_numpy(
        x_value,
        times[score_indices],
        joint_count=joint_count,
        harmonic_count=args.fourier_harmonics,
        base_period=args.base_period,
    )
    score = regressor_model.logdet_information(
        score_q,
        score_dq,
        score_ddq,
        ridge=args.ridge,
        condition_penalty=args.condition_penalty,
    )
    return times, q, dq, ddq, x_value, score


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    regressor_model: BaseRegressorModel,
    times: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    coefficients: np.ndarray,
    score: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tau = np.vstack([regressor_model.inverse_dynamics(q_i, dq_i, ddq_i) for q_i, dq_i, ddq_i in zip(q, dq, ddq)])
    np.savez_compressed(
        output_dir / "trajectory.npz",
        times=times,
        positions=q,
        velocities=dq,
        accelerations=ddq,
        torques=tau,
        coefficients=coefficients,
        joint_names=np.asarray(FRANKA_JOINTS),
    )

    header = (
        ["time"]
        + [f"{name}.position" for name in FRANKA_JOINTS]
        + [f"{name}.velocity" for name in FRANKA_JOINTS]
        + [f"{name}.acceleration" for name in FRANKA_JOINTS]
        + [f"{name}.torque_preview" for name in FRANKA_JOINTS]
    )
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in zip(times, q, dq, ddq, tau):
            writer.writerow([float(row[0]), *row[1].tolist(), *row[2].tolist(), *row[3].tolist(), *row[4].tolist()])

    trajectory_json = {
        "schema": "franka_sysid_offline_fourier_trajectory_v1",
        "joint_names": FRANKA_JOINTS,
        "sample_rate_hz": args.sample_rate,
        "base_period_sec": args.base_period,
        "cycles": args.cycles,
        "points": [
            {
                "time": float(t),
                "positions": q_i.tolist(),
                "velocities": dq_i.tolist(),
                "accelerations": ddq_i.tolist(),
                "torque_preview": tau_i.tolist(),
            }
            for t, q_i, dq_i, ddq_i, tau_i in zip(times, q, dq, ddq, tau)
        ],
    }
    (output_dir / "trajectory.json").write_text(json.dumps(trajectory_json, indent=2), encoding="utf-8")

    min_q = np.asarray([limit[0] for limit in FRANKA_LIMITS], dtype=np.float64)
    max_q = np.asarray([limit[1] for limit in FRANKA_LIMITS], dtype=np.float64)
    manifest = {
        "schema": "franka_sysid_offline_d_optimal_run_v1",
        "created_wall_time": time.time(),
        "urdf_path": str(Path(args.urdf_path).expanduser().resolve()),
        "joint_names": FRANKA_JOINTS,
        "d_optimal_score": float(score),
        "base_regressor_rank": regressor_model.structural_rank,
        "full_parameter_count": regressor_model.full_parameter_count,
        "base_columns": regressor_model.base_columns.tolist(),
        "limits": {
            "position_min": min_q.tolist(),
            "position_max": max_q.tolist(),
            "max_velocity": args.max_joint_velocity,
            "max_acceleration": args.max_joint_acceleration,
        },
        "peaks": {
            "abs_position": np.max(np.abs(q), axis=0).tolist(),
            "abs_velocity": np.max(np.abs(dq), axis=0).tolist(),
            "abs_acceleration": np.max(np.abs(ddq), axis=0).tolist(),
            "abs_torque_preview": np.max(np.abs(tau), axis=0).tolist(),
        },
        "solver": {
            "fourier_harmonics": args.fourier_harmonics,
            "seed": args.seed,
            "score_stride": args.score_stride,
            "ridge": args.ridge,
            "condition_penalty": args.condition_penalty,
            "ipopt_max_iter": args.ipopt_max_iter,
            "ipopt_tolerance": args.ipopt_tolerance,
        },
        "files": {
            "npz": "trajectory.npz",
            "csv": "trajectory.csv",
            "json": "trajectory.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.no_plots:
        write_plots(output_dir, times, q, dq, ddq, tau)


def write_plots(output_dir: Path, times: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, tau: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    series = [
        ("positions", q, "rad"),
        ("velocities", dq, "rad/s"),
        ("accelerations", ddq, "rad/s^2"),
        ("torque_preview", tau, "Nm"),
    ]
    for name, values, ylabel in series:
        fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for joint_i, axis in enumerate(axes):
            axis.plot(times, values[:, joint_i])
            axis.set_ylabel(f"j{joint_i + 1} {ylabel}")
            axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--base-period", type=float, default=8.0)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--amplitude-scale", type=float, default=0.70)
    parser.add_argument("--fourier-harmonics", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--score-stride", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--condition-penalty", type=float, default=0.0)
    parser.add_argument("--max-joint-velocity", type=float, default=0.65)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.50)
    parser.add_argument("--base-regressor-samples", type=int, default=240)
    parser.add_argument("--base-regressor-rank-tolerance", type=float, default=1e-8)
    parser.add_argument("--ipopt-max-iter", type=int, default=500)
    parser.add_argument("--ipopt-print-level", type=int, default=5)
    parser.add_argument("--ipopt-tolerance", type=float, default=1e-6)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fourier_harmonics < 1:
        print("--fourier-harmonics must be >= 1", file=sys.stderr)
        return 2
    if args.base_regressor_samples < 2:
        print("--base-regressor-samples must be >= 2", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"franka_sysid_offline_v3_{stamp}").expanduser().resolve()
    regressor_model = load_base_regressor_model(
        urdf_path=args.urdf_path,
        joint_names=FRANKA_JOINTS,
        center=FRANKA_CENTER,
        amplitudes=(args.amplitude_scale * np.asarray(FRANKA_AMPLITUDES, dtype=np.float64)).tolist(),
        base_period=args.base_period,
        structural_samples=args.base_regressor_samples,
        rank_tolerance=args.base_regressor_rank_tolerance,
    )
    times, q, dq, ddq, coefficients, score = solve_offline(args, regressor_model)
    write_outputs(output_dir, args, regressor_model, times, q, dq, ddq, coefficients, score)
    print(f"Wrote offline trajectory package: {output_dir}")
    print(f"D-optimal score: {score:.6g}")
    print(
        f"Base regressor rank: {regressor_model.structural_rank}/"
        f"{regressor_model.full_parameter_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
