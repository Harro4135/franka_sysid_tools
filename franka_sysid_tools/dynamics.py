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
class RegressorDiagnostics:
    """Rank and spectrum summary for a stacked regressor matrix."""

    row_count: int
    column_count: int
    rank: int
    rank_tolerance: float
    rank_cutoff: float
    singular_values: np.ndarray
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    logdet_regularized: float
    ridge: float

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "rank": self.rank,
            "rank_tolerance": self.rank_tolerance,
            "rank_cutoff": self.rank_cutoff,
            "singular_values": self.singular_values.tolist(),
            "min_singular_value": self.min_singular_value,
            "max_singular_value": self.max_singular_value,
            "condition_number": self.condition_number,
            "logdet_regularized": self.logdet_regularized,
            "ridge": self.ridge,
        }


@dataclass
class BaseRegressorModel:
    """Pinocchio torque regressor plus selected identifiable base columns."""

    model: object
    data: object
    joint_names: list[str]
    base_columns: np.ndarray
    structural_rank: int
    full_parameter_count: int
    structural_diagnostics: RegressorDiagnostics | None = None
    structural_sampling: str = "trajectory"
    rejected_columns: np.ndarray | None = None

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

    def friction_regressor(self, dq: np.ndarray) -> np.ndarray:
        dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        return np.hstack((np.diag(np.sign(dq)), np.diag(dq)))

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

    def stacked_full_regressor(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        include_friction: bool = False,
    ) -> np.ndarray:
        rows = []
        for q_i, dq_i, ddq_i in zip(q, dq, ddq):
            row = self.full_regressor(q_i, dq_i, ddq_i)
            if include_friction:
                row = np.hstack((row, self.friction_regressor(dq_i)))
            rows.append(row)
        return np.vstack(rows)

    def stacked_base_regressor(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        include_friction: bool = False,
    ) -> np.ndarray:
        rows = []
        for q_i, dq_i, ddq_i in zip(q, dq, ddq):
            row = self.base_regressor(q_i, dq_i, ddq_i)
            if include_friction:
                row = np.hstack((row, self.friction_regressor(dq_i)))
            rows.append(row)
        return np.vstack(rows)

    @staticmethod
    def information_matrix(regressor: np.ndarray, *, ridge: float = 0.0) -> np.ndarray:
        regressor = np.asarray(regressor, dtype=np.float64)
        information = regressor.T @ regressor
        if ridge > 0.0:
            information = information + float(ridge) * np.eye(information.shape[0])
        return information

    @staticmethod
    def diagnostics_for_matrix(
        matrix: np.ndarray,
        *,
        rank_tolerance: float,
        ridge: float,
    ) -> RegressorDiagnostics:
        matrix = np.asarray(matrix, dtype=np.float64)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size:
            max_sv = float(singular_values[0])
            min_sv = float(singular_values[-1])
        else:
            max_sv = 0.0
            min_sv = 0.0
        rank_cutoff = float(rank_tolerance) * max(max_sv, 1.0)
        rank = int(np.sum(singular_values > rank_cutoff))
        if min_sv <= 0.0:
            condition_number = float("inf") if max_sv > 0.0 else 0.0
        else:
            condition_number = float(max_sv / min_sv)
        information = BaseRegressorModel.information_matrix(matrix, ridge=float(ridge))
        sign, logdet = np.linalg.slogdet(information)
        return RegressorDiagnostics(
            row_count=int(matrix.shape[0]),
            column_count=int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            rank=rank,
            rank_tolerance=float(rank_tolerance),
            rank_cutoff=float(rank_cutoff),
            singular_values=singular_values,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            condition_number=condition_number,
            logdet_regularized=float(logdet) if sign > 0 else float("-inf"),
            ridge=float(ridge),
        )

    def trajectory_diagnostics(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        rank_tolerance: float,
        ridge: float,
        include_friction: bool = False,
        base_only: bool = True,
    ) -> RegressorDiagnostics:
        if base_only:
            matrix = self.stacked_base_regressor(q, dq, ddq, include_friction=include_friction)
        else:
            matrix = self.stacked_full_regressor(q, dq, ddq, include_friction=include_friction)
        return self.diagnostics_for_matrix(matrix, rank_tolerance=rank_tolerance, ridge=ridge)

    def logdet_information(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        ridge: float,
        condition_penalty: float = 0.0,
        include_friction: bool = False,
    ) -> float:
        return self.objective_score(
            q,
            dq,
            ddq,
            ridge=ridge,
            condition_penalty=condition_penalty,
            objective="conditioned_d_opt" if condition_penalty > 0.0 else "d_opt",
            include_friction=include_friction,
        )

    def objective_score(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        *,
        ridge: float,
        condition_penalty: float,
        objective: str,
        include_friction: bool,
    ) -> float:
        w_base = self.stacked_base_regressor(q, dq, ddq, include_friction=include_friction)
        information = self.information_matrix(w_base, ridge=float(ridge))
        eigenvalues = np.linalg.eigvalsh(information)
        if eigenvalues.size == 0:
            return float("-inf")
        min_eig = float(max(eigenvalues[0], 1e-300))
        max_eig = float(max(eigenvalues[-1], min_eig))
        condition = max_eig / min_eig
        logdet = float(np.sum(np.log(np.maximum(eigenvalues, 1e-300))))
        if objective == "d_opt":
            return logdet
        if objective == "conditioned_d_opt":
            return float(logdet - float(condition_penalty) * np.log(max(condition, 1.0)))
        if objective == "e_opt":
            return float(np.log(min_eig))
        if objective == "condition":
            return float(-np.log(max(condition, 1.0)))
        raise ValueError(f"Unsupported objective: {objective}")


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


def _random_feasible_samples(
    *,
    center: np.ndarray,
    amplitudes: np.ndarray,
    joint_limits: np.ndarray,
    max_joint_velocity: float,
    max_joint_acceleration: float,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    lower = np.maximum(joint_limits[:, 0], center - amplitudes)
    upper = np.minimum(joint_limits[:, 1], center + amplitudes)
    if np.any(lower > upper):
        raise ValueError("Random feasible sampling envelope has lower bounds above upper bounds")
    count = int(sample_count)
    q = rng.uniform(lower, upper, size=(count, center.size))
    dq = rng.uniform(-float(max_joint_velocity), float(max_joint_velocity), size=q.shape)
    ddq = rng.uniform(-float(max_joint_acceleration), float(max_joint_acceleration), size=q.shape)
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
    structural_sampling: str = "trajectory",
    seed: int = 0,
    joint_limits: list[tuple[float, float]] | None = None,
    max_joint_velocity: float | None = None,
    max_joint_acceleration: float | None = None,
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
    center_array = np.asarray(center, dtype=np.float64)
    amplitude_array = np.asarray(amplitudes, dtype=np.float64)
    if structural_sampling == "random_feasible":
        if joint_limits is None or max_joint_velocity is None or max_joint_acceleration is None:
            raise ValueError(
                "random_feasible base-regressor sampling requires joint_limits, "
                "max_joint_velocity, and max_joint_acceleration"
            )
        q, dq, ddq = _random_feasible_samples(
            center=center_array,
            amplitudes=amplitude_array,
            joint_limits=np.asarray(joint_limits, dtype=np.float64),
            max_joint_velocity=float(max_joint_velocity),
            max_joint_acceleration=float(max_joint_acceleration),
            sample_count=structural_samples,
            seed=int(seed),
        )
    elif structural_sampling == "trajectory":
        q, dq, ddq = _structural_trajectory(
            center=center_array,
            amplitudes=amplitude_array,
            base_period=base_period,
            sample_count=structural_samples,
        )
    else:
        raise ValueError(f"Unsupported base-regressor sampling mode: {structural_sampling}")
    full_rows = [
        np.asarray(pin.computeJointTorqueRegressor(model, data, q_i, dq_i, ddq_i), dtype=np.float64)
        for q_i, dq_i, ddq_i in zip(q, dq, ddq)
    ]
    full = np.vstack(full_rows)
    structural_diagnostics = BaseRegressorModel.diagnostics_for_matrix(
        full,
        rank_tolerance=rank_tolerance,
        ridge=0.0,
    )
    singular_values = structural_diagnostics.singular_values
    if singular_values.size == 0:
        raise RuntimeError("Pinocchio torque regressor produced an empty structural matrix")
    structural_rank = structural_diagnostics.rank
    base_columns = np.asarray(_greedy_independent_columns(full, rank_tolerance, structural_rank), dtype=np.int64)
    if base_columns.size == 0:
        raise RuntimeError("Could not identify any independent columns in the Pinocchio torque regressor")
    if base_columns.size < structural_rank:
        raise RuntimeError(
            f"Base-column selection found {base_columns.size} columns, below SVD structural rank {structural_rank}"
        )
    rejected_columns = np.setdiff1d(np.arange(full.shape[1], dtype=np.int64), base_columns, assume_unique=False)
    return BaseRegressorModel(
        model=model,
        data=data,
        joint_names=list(joint_names),
        base_columns=base_columns,
        structural_rank=int(structural_rank),
        full_parameter_count=int(full.shape[1]),
        structural_diagnostics=structural_diagnostics,
        structural_sampling=str(structural_sampling),
        rejected_columns=rejected_columns,
    )
