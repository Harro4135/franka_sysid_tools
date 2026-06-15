"""Dynamics-regressor helpers for SysID trajectory design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "franka_sysid_collect_v3 requires Pinocchio for physical regressor scoring. "
            "Install python3-pinocchio in the ROS environment, or provide an environment "
            "where `import pinocchio` succeeds."
        ) from exc
    return pin


def _greedy_independent_columns(matrix: np.ndarray, tolerance: float, target_rank: int) -> list[int]:
    """Rank-revealing column selection without requiring SciPy's pivoted QR."""

    residual = np.asarray(matrix, dtype=np.float64).copy()
    selected: list[int] = []
    original_norm = float(np.linalg.norm(residual, ord="fro"))
    cutoff = max(float(tolerance) * max(original_norm, 1.0), 1e-12)
    while len(selected) < int(target_rank) and residual.shape[1] and float(np.max(np.linalg.norm(residual, axis=0))) > cutoff:
        norms = np.linalg.norm(residual, axis=0)
        pivot = int(np.argmax(norms))
        column = residual[:, pivot].copy()
        norm = float(np.linalg.norm(column))
        if norm <= cutoff:
            break
        selected.append(pivot)
        basis = column / norm
        residual -= np.outer(basis, basis @ residual)
        residual[:, pivot] = 0.0
    return selected


@dataclass
class BaseRegressorModel:
    """Pinocchio torque regressor plus selected identifiable base columns."""

    model: object
    data: object
    joint_names: list[str]
    base_columns: np.ndarray
    structural_rank: int
    full_parameter_count: int

    def full_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        pin = _import_pinocchio()
        return np.asarray(
            pin.computeJointTorqueRegressor(
                self.model,
                self.data,
                np.asarray(q, dtype=np.float64),
                np.asarray(dq, dtype=np.float64),
                np.asarray(ddq, dtype=np.float64),
            ),
            dtype=np.float64,
        )

    def base_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        return self.full_regressor(q, dq, ddq)[:, self.base_columns]

    def inverse_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        pin = _import_pinocchio()
        return np.asarray(
            pin.rnea(
                self.model,
                self.data,
                np.asarray(q, dtype=np.float64),
                np.asarray(dq, dtype=np.float64),
                np.asarray(ddq, dtype=np.float64),
            ),
            dtype=np.float64,
        )

    def stacked_base_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        rows = [self.base_regressor(q_i, dq_i, ddq_i) for q_i, dq_i, ddq_i in zip(q, dq, ddq)]
        return np.vstack(rows)

    def logdet_information(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        ridge: float,
        condition_penalty: float = 0.0,
    ) -> float:
        w_base = self.stacked_base_regressor(q, dq, ddq)
        information = w_base.T @ w_base + float(ridge) * np.eye(w_base.shape[1])
        sign, logdet = np.linalg.slogdet(information)
        if sign <= 0:
            return float("-inf")
        if condition_penalty <= 0.0:
            return float(logdet)
        singular_values = np.linalg.svd(information, compute_uv=False)
        condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
        return float(logdet - float(condition_penalty) * np.log(condition))


def _structural_trajectory(
    *,
    center: np.ndarray,
    amplitudes: np.ndarray,
    base_period: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, float(base_period), int(sample_count))
    omega = 2.0 * np.pi / float(base_period)
    q = np.zeros((len(times), len(center)), dtype=np.float64)
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    for joint_i in range(len(center)):
        h1 = 1 + joint_i % 4
        h2 = 2 + joint_i % 5
        phase = 0.37 * joint_i
        q[:, joint_i] = center[joint_i] + amplitudes[joint_i] * (
            0.65 * np.sin(h1 * omega * times + phase) + 0.35 * np.cos(h2 * omega * times - phase)
        )
        dq[:, joint_i] = amplitudes[joint_i] * (
            0.65 * h1 * omega * np.cos(h1 * omega * times + phase)
            - 0.35 * h2 * omega * np.sin(h2 * omega * times - phase)
        )
        ddq[:, joint_i] = amplitudes[joint_i] * (
            -0.65 * (h1 * omega) ** 2 * np.sin(h1 * omega * times + phase)
            - 0.35 * (h2 * omega) ** 2 * np.cos(h2 * omega * times - phase)
        )
    return q, dq, ddq


def load_base_regressor_model(
    *,
    urdf_path: str | Path,
    joint_names: list[str],
    center: list[float],
    amplitudes: list[float],
    base_period: float,
    structural_samples: int,
    rank_tolerance: float,
) -> BaseRegressorModel:
    pin = _import_pinocchio()
    path = Path(urdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"URDF path does not exist: {path}")

    model = pin.buildModelFromUrdf(str(path))
    if int(model.nv) != len(joint_names):
        raise ValueError(
            f"Pinocchio model has nv={model.nv}, but collector expects {len(joint_names)} joints: {joint_names}. "
            "Use a fixed-base Panda arm URDF matching the controlled joint list."
        )
    data = model.createData()
    q, dq, ddq = _structural_trajectory(
        center=np.asarray(center, dtype=np.float64),
        amplitudes=np.asarray(amplitudes, dtype=np.float64),
        base_period=base_period,
        sample_count=structural_samples,
    )
    full_rows = [
        np.asarray(pin.computeJointTorqueRegressor(model, data, q_i, dq_i, ddq_i), dtype=np.float64)
        for q_i, dq_i, ddq_i in zip(q, dq, ddq)
    ]
    full = np.vstack(full_rows)
    singular_values = np.linalg.svd(full, compute_uv=False)
    if singular_values.size == 0:
        raise RuntimeError("Pinocchio torque regressor produced an empty structural matrix")
    rank_cutoff = float(rank_tolerance) * max(float(singular_values[0]), 1.0)
    structural_rank = int(np.sum(singular_values > rank_cutoff))
    base_columns = np.asarray(_greedy_independent_columns(full, rank_tolerance, structural_rank), dtype=np.int64)
    if base_columns.size == 0:
        raise RuntimeError("Could not identify any independent columns in the Pinocchio torque regressor")
    if base_columns.size < structural_rank:
        raise RuntimeError(
            f"Base-column selection found {base_columns.size} columns, below SVD structural rank {structural_rank}"
        )
    return BaseRegressorModel(
        model=model,
        data=data,
        joint_names=list(joint_names),
        base_columns=base_columns,
        structural_rank=int(structural_rank),
        full_parameter_count=int(full.shape[1]),
    )
