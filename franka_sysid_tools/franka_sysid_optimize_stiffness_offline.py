"""Fast, position-domain trajectory design for Franka drive-stiffness SysID.

Unlike the rigid-body v3 designer, this module does not construct an inertial
regressor and does not require Pinocchio, CasADi, ROS, or a URDF.  It chooses a
periodic Fourier command by maximizing the predicted sensitivity of a
second-order closed-loop joint response to log(stiffness), robustly averaged
over a user-supplied natural-frequency range.

The output trajectory JSON is intentionally compatible with
``franka_sysid_collect_v3``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FRANKA_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_CENTER = np.asarray([0.0, -0.75, 0.0, -2.20, 0.0, 1.75, 0.80], dtype=np.float64)
# Nominal per-joint excursion allowances.  The base receives the largest
# increase; the other joints are raised modestly while retaining extra margin
# on the elbow/wrist joints whose centers sit closer to their useful envelope.
FRANKA_AMPLITUDES = np.asarray([0.40, 0.28, 0.38, 0.25, 0.38, 0.25, 0.38], dtype=np.float64)
FRANKA_LIMITS = np.asarray(
    [
        (-2.70, 2.70),
        (-1.55, 1.55),
        (-2.70, 2.70),
        (-2.95, -0.25),
        (-2.70, 2.70),
        (0.15, 3.45),
        (-2.70, 2.70),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DesignConfig:
    base_period: float
    cycles: int
    harmonic_count: int
    sample_rate: float
    constraint_rate: float
    amplitude_scale: float
    max_joint_velocity: float
    max_joint_acceleration: float
    max_joint_jerk: float
    natural_frequency_min_hz: float
    natural_frequency_max_hz: float
    damping_ratio: float
    bandwidth_samples: int
    candidate_count: int
    seed: int
    correlation_penalty: float
    damping_target: str
    repeat_cycles: bool = True


@dataclass
class JointDesign:
    sin_coefficients: np.ndarray
    cos_coefficients: np.ndarray
    predicted_information: float
    unconstrained_score: float
    scale: float
    active_constraint: str
    peak_position_offset: float
    peak_velocity: float
    peak_acceleration: float
    peak_jerk: float
    command_wave: np.ndarray


def _project_onto_null(vector: np.ndarray, constraint: np.ndarray) -> np.ndarray:
    """Project ``vector`` so ``constraint @ vector == 0``."""

    denom = float(constraint @ constraint)
    if denom <= 0.0:
        raise ValueError("projection constraint must be nonzero")
    return vector - constraint * float(constraint @ vector) / denom


def stiffness_sensitivity_gain_squared(
    frequencies_hz: np.ndarray,
    natural_frequencies_hz: np.ndarray,
    *,
    damping_ratio: float,
    damping_target: str = "reference",
) -> np.ndarray:
    """Return |d q / d log(K)|^2 per unit sinusoidal position command.

    Each joint is represented by ``M qdd = K(q_ref-q) + D(v_target-dq)``.
    ``damping_target='reference'`` uses the commanded trajectory velocity;
    ``'zero'`` models a damper acting only against measured joint velocity.
    The derivative holds M and D fixed while perturbing K.
    """

    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1, 1)
    natural_frequencies_hz = np.asarray(natural_frequencies_hz, dtype=np.float64).reshape(1, -1)
    if np.any(frequencies_hz <= 0.0) or np.any(natural_frequencies_hz <= 0.0):
        raise ValueError("frequencies must be positive")
    if damping_ratio <= 0.0:
        raise ValueError("damping_ratio must be positive")
    if damping_target not in {"reference", "zero"}:
        raise ValueError("damping_target must be 'reference' or 'zero'")

    omega = 2.0 * math.pi * frequencies_hz
    omega_n = 2.0 * math.pi * natural_frequencies_hz
    damping_over_mass = 2.0 * float(damping_ratio) * omega_n
    denominator = omega_n**2 - omega**2 + 1j * damping_over_mass * omega

    if damping_target == "reference":
        # H=(K+jDw)/(K-Mw^2+jDw), with M normalized to one.
        derivative = -(omega_n**2) * omega**2 / denominator**2
    else:
        # H=K/(K-Mw^2+jDw), again perturbing K while holding M,D fixed.
        derivative = (omega_n**2) * (-omega**2 + 1j * damping_over_mass * omega) / denominator**2
    return np.abs(derivative) ** 2


def _harmonic_layout(config: DesignConfig) -> tuple[np.ndarray, float]:
    """Return integer Fourier bins and the period over which they are defined."""

    if config.repeat_cycles or config.cycles == 1:
        return np.arange(1, config.harmonic_count + 1, dtype=np.float64), config.base_period

    # Preserve the original frequency neighborhood, but move interior bins off
    # exact multiples of ``cycles``.  Their gcd is one, so the combined command
    # has one full-duration period instead of repeating every base period.
    indices = config.cycles * np.arange(1, config.harmonic_count + 1, dtype=np.int64)
    if config.cycles == 2:
        indices[1::2] += 1
    elif config.harmonic_count == 2:
        indices[1] += 1
    else:
        for index in range(1, config.harmonic_count - 1):
            indices[index] += 1 if index % 2 else -1

    if np.any(np.diff(indices) <= 0):
        raise RuntimeError(f"could not construct increasing nonrepeating Fourier bins: {indices.tolist()}")
    common_divisor = 0
    for index in indices:
        common_divisor = math.gcd(common_divisor, int(index))
    if common_divisor != 1:
        raise RuntimeError(f"nonrepeating Fourier bins have gcd {common_divisor}: {indices.tolist()}")
    return indices.astype(np.float64), config.base_period * config.cycles


def robust_harmonic_gains(
    config: DesignConfig,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    harmonic_indices, spectral_period = _harmonic_layout(config)
    harmonic_frequencies = harmonic_indices / spectral_period
    natural_frequencies = np.geomspace(
        config.natural_frequency_min_hz,
        config.natural_frequency_max_hz,
        config.bandwidth_samples,
    )
    gain_grid = stiffness_sensitivity_gain_squared(
        harmonic_frequencies,
        natural_frequencies,
        damping_ratio=config.damping_ratio,
        damping_target=config.damping_target,
    )
    # A geometric mean is deliberately less resonance-dominated than an
    # arithmetic mean and rewards excitation useful across the whole band.
    robust_gain = np.exp(np.mean(np.log(gain_grid + 1e-18), axis=1))
    return harmonic_indices, spectral_period, harmonic_frequencies, natural_frequencies, robust_gain


def evaluate_fourier(
    times: np.ndarray,
    centers: np.ndarray,
    sin_coefficients: np.ndarray,
    cos_coefficients: np.ndarray,
    *,
    base_period: float,
    harmonic_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate position Fourier coefficients and their first 3 derivatives."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    sin_coefficients = np.asarray(sin_coefficients, dtype=np.float64)
    cos_coefficients = np.asarray(cos_coefficients, dtype=np.float64)
    if sin_coefficients.shape != cos_coefficients.shape:
        raise ValueError("sine and cosine coefficient shapes differ")
    if sin_coefficients.shape[0] != len(centers):
        raise ValueError("coefficient joint count differs from center count")

    if harmonic_indices is None:
        harmonics = np.arange(1, sin_coefficients.shape[1] + 1, dtype=np.float64)
    else:
        harmonics = np.asarray(harmonic_indices, dtype=np.float64).reshape(-1)
        if harmonics.shape != (sin_coefficients.shape[1],):
            raise ValueError("harmonic index count differs from coefficient count")
    omega_h = (2.0 * math.pi / float(base_period)) * harmonics
    phase = times[:, np.newaxis] * omega_h[np.newaxis, :]
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    q = centers[np.newaxis, :] + sin_phase @ sin_coefficients.T + cos_phase @ cos_coefficients.T
    dq = (cos_phase * omega_h) @ sin_coefficients.T - (sin_phase * omega_h) @ cos_coefficients.T
    ddq = -(sin_phase * omega_h**2) @ sin_coefficients.T - (cos_phase * omega_h**2) @ cos_coefficients.T
    jerk = -(cos_phase * omega_h**3) @ sin_coefficients.T + (sin_phase * omega_h**3) @ cos_coefficients.T
    return q, dq, ddq, jerk


def _candidate_scale(
    position_offset: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    *,
    center: float,
    target_amplitude: float,
    lower_limit: float,
    upper_limit: float,
    config: DesignConfig,
) -> tuple[float, str]:
    factors: list[tuple[str, float]] = []

    peak_position = float(np.max(np.abs(position_offset)))
    peak_velocity = float(np.max(np.abs(velocity)))
    peak_acceleration = float(np.max(np.abs(acceleration)))
    peak_jerk = float(np.max(np.abs(jerk)))
    factors.append(("target_position_amplitude", target_amplitude / max(peak_position, 1e-15)))
    factors.append(("velocity", config.max_joint_velocity / max(peak_velocity, 1e-15)))
    factors.append(("acceleration", config.max_joint_acceleration / max(peak_acceleration, 1e-15)))
    if config.max_joint_jerk > 0.0:
        factors.append(("jerk", config.max_joint_jerk / max(peak_jerk, 1e-15)))

    positive_peak = float(np.max(position_offset))
    negative_peak = float(np.min(position_offset))
    if positive_peak > 0.0:
        factors.append(("upper_position_limit", (upper_limit - center) / positive_peak))
    if negative_peak < 0.0:
        factors.append(("lower_position_limit", (center - lower_limit) / -negative_peak))

    active_constraint, scale = min(factors, key=lambda pair: pair[1])
    if scale <= 0.0 or not math.isfinite(scale):
        raise RuntimeError("trajectory center or limits leave no positive feasible scale")
    # Reserve a small numerical margin between dense-grid verification and the
    # continuous command extrema.
    return 0.995 * float(scale), active_constraint


def _design_joint(
    joint_index: int,
    config: DesignConfig,
    robust_gain: np.ndarray,
    constraint_times: np.ndarray,
    previous_waves: list[np.ndarray],
    harmonic_indices: np.ndarray,
    spectral_period: float,
) -> JointDesign:
    harmonic_count = config.harmonic_count
    harmonics = np.asarray(harmonic_indices, dtype=np.float64)
    omega_h = (2.0 * math.pi / spectral_period) * harmonics
    phase = constraint_times[:, np.newaxis] * omega_h[np.newaxis, :]
    sin_basis = np.sin(phase)
    cos_basis = np.cos(phase)
    velocity_sin_basis = cos_basis * omega_h
    velocity_cos_basis = -sin_basis * omega_h
    acceleration_sin_basis = -sin_basis * omega_h**2
    acceleration_cos_basis = -cos_basis * omega_h**2
    jerk_sin_basis = -cos_basis * omega_h**3
    jerk_cos_basis = sin_basis * omega_h**3

    normalized_gain = robust_gain / max(float(np.max(robust_gain)), 1e-18)
    proposal_weight = 0.15 + np.sqrt(normalized_gain)
    rng = np.random.default_rng(config.seed + 1009 * joint_index)
    best: JointDesign | None = None
    best_adjusted_score = -math.inf

    for candidate_index in range(config.candidate_count):
        if candidate_index == 0:
            raw_sin = proposal_weight * np.where((np.arange(harmonic_count) + joint_index) % 2, -1.0, 1.0)
            raw_cos = proposal_weight * np.roll(raw_sin, 1)
        else:
            raw_sin = proposal_weight * rng.normal(size=harmonic_count)
            raw_cos = proposal_weight * rng.normal(size=harmonic_count)

        sin_coeff = _project_onto_null(raw_sin, harmonics)
        cos_coeff = _project_onto_null(raw_cos, harmonics**2)
        coefficient_norm = float(np.linalg.norm(np.concatenate((sin_coeff, cos_coeff))))
        if coefficient_norm < 1e-10:
            continue
        sin_coeff /= coefficient_norm
        cos_coeff /= coefficient_norm

        position_offset = sin_basis @ sin_coeff + cos_basis @ cos_coeff
        velocity = velocity_sin_basis @ sin_coeff + velocity_cos_basis @ cos_coeff
        acceleration = acceleration_sin_basis @ sin_coeff + acceleration_cos_basis @ cos_coeff
        jerk = jerk_sin_basis @ sin_coeff + jerk_cos_basis @ cos_coeff
        scale, active_constraint = _candidate_scale(
            position_offset,
            velocity,
            acceleration,
            jerk,
            center=float(FRANKA_CENTER[joint_index]),
            target_amplitude=float(config.amplitude_scale * FRANKA_AMPLITUDES[joint_index]),
            lower_limit=float(FRANKA_LIMITS[joint_index, 0]),
            upper_limit=float(FRANKA_LIMITS[joint_index, 1]),
            config=config,
        )
        sin_scaled = scale * sin_coeff
        cos_scaled = scale * cos_coeff
        wave = scale * position_offset
        component_energy = sin_scaled**2 + cos_scaled**2
        predicted_information = (
            0.5 * config.base_period * config.cycles * float(np.sum(robust_gain * component_energy))
        )
        unconstrained_score = float(np.sum(robust_gain * (sin_coeff**2 + cos_coeff**2)))
        correlation_cost = 0.0
        wave_norm = float(np.linalg.norm(wave))
        for previous in previous_waves:
            denom = wave_norm * float(np.linalg.norm(previous))
            if denom > 1e-15:
                correlation_cost += (float(wave @ previous) / denom) ** 2
        adjusted_score = math.log(predicted_information + 1e-18) - config.correlation_penalty * correlation_cost

        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best = JointDesign(
                sin_coefficients=sin_scaled,
                cos_coefficients=cos_scaled,
                predicted_information=predicted_information,
                unconstrained_score=unconstrained_score,
                scale=scale,
                active_constraint=active_constraint,
                peak_position_offset=float(np.max(np.abs(wave))),
                peak_velocity=float(np.max(np.abs(scale * velocity))),
                peak_acceleration=float(np.max(np.abs(scale * acceleration))),
                peak_jerk=float(np.max(np.abs(scale * jerk))),
                command_wave=wave,
            )

    if best is None:
        raise RuntimeError(f"could not construct a feasible candidate for {FRANKA_JOINTS[joint_index]}")
    return best


def design_trajectory(config: DesignConfig) -> dict[str, object]:
    """Design all seven joint commands and return sampled arrays plus metrics."""

    _validate_config(config)
    harmonic_indices, spectral_period, harmonic_frequencies, natural_frequencies, robust_gain = (
        robust_harmonic_gains(config)
    )
    constraint_count = max(1001, int(math.ceil(spectral_period * config.constraint_rate)) + 1)
    constraint_times = np.linspace(0.0, spectral_period, constraint_count)

    joint_designs: list[JointDesign] = []
    previous_waves: list[np.ndarray] = []
    for joint_index in range(len(FRANKA_JOINTS)):
        design = _design_joint(
            joint_index,
            config,
            robust_gain,
            constraint_times,
            previous_waves,
            harmonic_indices,
            spectral_period,
        )
        joint_designs.append(design)
        previous_waves.append(design.command_wave)

    sin_coefficients = np.vstack([design.sin_coefficients for design in joint_designs])
    cos_coefficients = np.vstack([design.cos_coefficients for design in joint_designs])
    duration = config.cycles * config.base_period
    sample_count = max(2, int(round(duration * config.sample_rate)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    q, dq, ddq, jerk = evaluate_fourier(
        times,
        FRANKA_CENTER,
        sin_coefficients,
        cos_coefficients,
        base_period=spectral_period,
        harmonic_indices=harmonic_indices,
    )

    dense_q, dense_dq, dense_ddq, dense_jerk = evaluate_fourier(
        constraint_times,
        FRANKA_CENTER,
        sin_coefficients,
        cos_coefficients,
        base_period=spectral_period,
        harmonic_indices=harmonic_indices,
    )
    _verify_constraints(config, dense_q, dense_dq, dense_ddq, dense_jerk)

    command_correlation = np.corrcoef((dense_q - FRANKA_CENTER).T)
    max_cross_correlation = float(np.max(np.abs(command_correlation - np.eye(len(FRANKA_JOINTS)))))
    return {
        "times": times,
        "positions": q,
        "velocities": dq,
        "accelerations": ddq,
        "jerks": jerk,
        "sin_coefficients": sin_coefficients,
        "cos_coefficients": cos_coefficients,
        "harmonic_indices": harmonic_indices,
        "spectral_period_sec": spectral_period,
        "harmonic_frequencies_hz": harmonic_frequencies,
        "natural_frequency_samples_hz": natural_frequencies,
        "robust_gain_squared": robust_gain,
        "joint_designs": joint_designs,
        "max_cross_joint_command_correlation": max_cross_correlation,
    }


def _validate_config(config: DesignConfig) -> None:
    if config.base_period <= 0.0:
        raise ValueError("base_period must be positive")
    if config.cycles < 1:
        raise ValueError("cycles must be at least one")
    if config.harmonic_count < 2:
        raise ValueError("harmonic_count must be at least two to enforce zero boundary velocity/acceleration")
    if config.sample_rate <= 0.0 or config.constraint_rate <= 0.0:
        raise ValueError("sample and constraint rates must be positive")
    if not 0.0 < config.amplitude_scale <= 1.0:
        raise ValueError("amplitude_scale must be in (0, 1]")
    if config.max_joint_velocity <= 0.0 or config.max_joint_acceleration <= 0.0:
        raise ValueError("velocity and acceleration limits must be positive")
    if config.max_joint_jerk < 0.0:
        raise ValueError("max_joint_jerk cannot be negative")
    if not 0.0 < config.natural_frequency_min_hz <= config.natural_frequency_max_hz:
        raise ValueError("natural-frequency range is invalid")
    if config.bandwidth_samples < 2:
        raise ValueError("bandwidth_samples must be at least two")
    if config.candidate_count < 1:
        raise ValueError("candidate_count must be at least one")
    if config.correlation_penalty < 0.0:
        raise ValueError("correlation_penalty cannot be negative")


def _verify_constraints(
    config: DesignConfig,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    jerk: np.ndarray,
) -> None:
    tolerance = 1e-8
    if np.any(q < FRANKA_LIMITS[:, 0] - tolerance) or np.any(q > FRANKA_LIMITS[:, 1] + tolerance):
        raise RuntimeError("dense verification found a position-limit violation")
    if float(np.max(np.abs(dq))) > config.max_joint_velocity + tolerance:
        raise RuntimeError("dense verification found a velocity-limit violation")
    if float(np.max(np.abs(ddq))) > config.max_joint_acceleration + tolerance:
        raise RuntimeError("dense verification found an acceleration-limit violation")
    if config.max_joint_jerk > 0.0 and float(np.max(np.abs(jerk))) > config.max_joint_jerk + tolerance:
        raise RuntimeError("dense verification found a jerk-limit violation")
    if float(np.max(np.abs(dq[[0, -1], :]))) > 1e-8:
        raise RuntimeError("boundary velocity is not zero")
    if float(np.max(np.abs(ddq[[0, -1], :]))) > 1e-8:
        raise RuntimeError("boundary acceleration is not zero")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    times = np.asarray(result["times"])
    q = np.asarray(result["positions"])
    dq = np.asarray(result["velocities"])
    ddq = np.asarray(result["accelerations"])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["time_sec"]
        header.extend(f"{name}_position" for name in FRANKA_JOINTS)
        header.extend(f"{name}_velocity" for name in FRANKA_JOINTS)
        header.extend(f"{name}_acceleration" for name in FRANKA_JOINTS)
        writer.writerow(header)
        for index, time_sec in enumerate(times):
            writer.writerow([float(time_sec), *q[index].tolist(), *dq[index].tolist(), *ddq[index].tolist()])
    os.replace(temporary, path)


def _write_plots(output_dir: Path, result: dict[str, object]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    times = np.asarray(result["times"])
    plot_specs = (
        ("positions", np.asarray(result["positions"]), "Position [rad]"),
        ("velocities", np.asarray(result["velocities"]), "Velocity [rad/s]"),
        ("accelerations", np.asarray(result["accelerations"]), "Acceleration [rad/s^2]"),
    )
    written: list[str] = []
    for name, values, ylabel in plot_specs:
        figure, axis = plt.subplots(figsize=(12, 6))
        for joint_index, joint_name in enumerate(FRANKA_JOINTS):
            axis.plot(times, values[:, joint_index], label=joint_name)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4, fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(figure)
        written.append(f"{name}.png")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.semilogy(
        np.asarray(result["harmonic_frequencies_hz"]),
        np.asarray(result["robust_gain_squared"]),
        marker="o",
    )
    axis.set_xlabel("Command harmonic [Hz]")
    axis.set_ylabel("Robust |dq / d log(K)|^2 per command radian")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "stiffness_sensitivity.png", dpi=150)
    plt.close(figure)
    written.append("stiffness_sensitivity.png")
    return written


def write_outputs(output_dir: Path, config: DesignConfig, result: dict[str, object], elapsed_sec: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = output_dir / "_SUCCESS"
    if success_marker.exists():
        success_marker.unlink()

    times = np.asarray(result["times"])
    q = np.asarray(result["positions"])
    dq = np.asarray(result["velocities"])
    ddq = np.asarray(result["accelerations"])
    trajectory_payload = {
        "schema": "franka_sysid_offline_fourier_trajectory_v1",
        "design": "position_domain_stiffness_sensitivity_v1",
        "joint_names": FRANKA_JOINTS,
        "sample_rate_hz": config.sample_rate,
        "base_period_sec": config.base_period,
        "cycles": config.cycles,
        "repeat_cycles": config.repeat_cycles,
        "spectral_period_sec": float(result["spectral_period_sec"]),
        "harmonic_indices": np.asarray(result["harmonic_indices"]).astype(int).tolist(),
        "points": [
            {
                "time": float(times[index]),
                "positions": q[index].tolist(),
                "velocities": dq[index].tolist(),
                "accelerations": ddq[index].tolist(),
            }
            for index in range(len(times))
        ],
    }
    _atomic_write_text(output_dir / "trajectory.json", json.dumps(trajectory_payload, indent=2) + "\n")
    _write_csv(output_dir / "trajectory.csv", result)

    npz_temporary = output_dir / "trajectory.npz.tmp"
    with npz_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            time_sec=times,
            positions=q,
            velocities=dq,
            accelerations=ddq,
            jerks=np.asarray(result["jerks"]),
            sin_coefficients=np.asarray(result["sin_coefficients"]),
            cos_coefficients=np.asarray(result["cos_coefficients"]),
            harmonic_indices=np.asarray(result["harmonic_indices"]),
            spectral_period_sec=np.asarray(float(result["spectral_period_sec"])),
            joint_names=np.asarray(FRANKA_JOINTS),
        )
    os.replace(npz_temporary, output_dir / "trajectory.npz")
    plot_files = _write_plots(output_dir, result)

    joint_designs = result["joint_designs"]
    assert isinstance(joint_designs, list)
    manifest = {
        "schema": "franka_sysid_stiffness_design_manifest_v1",
        "status": "success",
        "design": "position_domain_stiffness_sensitivity_v1",
        "elapsed_sec": float(elapsed_sec),
        "config": asdict(config),
        "model": {
            "equation": "M*qdd = K*(q_ref-q) + D*(v_target-dq)",
            "identified_parameter": "per-joint drive stiffness K",
            "derivative_parameterization": "log(K), holding M and D fixed",
            "damping_target": config.damping_target,
            "repeat_cycles": config.repeat_cycles,
            "spectral_period_sec": float(result["spectral_period_sec"]),
            "harmonic_indices": np.asarray(result["harmonic_indices"]).astype(int).tolist(),
            "natural_frequency_samples_hz": np.asarray(result["natural_frequency_samples_hz"]).tolist(),
            "harmonic_frequencies_hz": np.asarray(result["harmonic_frequencies_hz"]).tolist(),
            "robust_gain_squared": np.asarray(result["robust_gain_squared"]).tolist(),
            "note": "This is a robust independent-joint design proxy; hardware collision preflight remains mandatory.",
        },
        "fourier": {
            "position_sine_coefficients": np.asarray(result["sin_coefficients"]).tolist(),
            "position_cosine_coefficients": np.asarray(result["cos_coefficients"]).tolist(),
        },
        "joint_metrics": {
            FRANKA_JOINTS[index]: {
                "predicted_information": float(design.predicted_information),
                "unconstrained_score": float(design.unconstrained_score),
                "scale": float(design.scale),
                "active_constraint": design.active_constraint,
                "peak_position_offset": float(design.peak_position_offset),
                "peak_velocity": float(design.peak_velocity),
                "peak_acceleration": float(design.peak_acceleration),
                "peak_jerk": float(design.peak_jerk),
            }
            for index, design in enumerate(joint_designs)
        },
        "verification": {
            "peak_velocity": float(np.max(np.abs(dq))),
            "peak_acceleration": float(np.max(np.abs(ddq))),
            "peak_jerk": float(np.max(np.abs(result["jerks"]))),
            "boundary_velocity": float(np.max(np.abs(dq[[0, -1], :]))),
            "boundary_acceleration": float(np.max(np.abs(ddq[[0, -1], :]))),
            "max_cross_joint_command_correlation": float(result["max_cross_joint_command_correlation"]),
        },
        "outputs": ["trajectory.json", "trajectory.csv", "trajectory.npz", *plot_files],
    }
    _atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_write_text(success_marker, "success\n")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Design a fast Fourier trajectory specifically for position-domain drive-stiffness identification."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-period", type=float, default=6.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument(
        "--nonrepeating",
        action="store_true",
        help=(
            "Use one full-duration Fourier realization with no repeated base-period cycles; "
            "zero velocity and acceleration are enforced only at the overall endpoints."
        ),
    )
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--constraint-rate", type=float, default=500.0)
    parser.add_argument("--amplitude-scale", type=float, default=0.90)
    parser.add_argument("--max-joint-velocity", type=float, default=0.85)
    parser.add_argument("--max-joint-acceleration", type=float, default=2.0)
    parser.add_argument(
        "--max-joint-jerk",
        type=float,
        default=0.0,
        help="Optional rad/s^3 constraint; zero disables it.",
    )
    parser.add_argument("--natural-frequency-min-hz", type=float, default=0.4)
    parser.add_argument("--natural-frequency-max-hz", type=float, default=2.0)
    parser.add_argument("--damping-ratio", type=float, default=0.7)
    parser.add_argument("--bandwidth-samples", type=int, default=9)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--correlation-penalty", type=float, default=0.25)
    parser.add_argument(
        "--damping-target",
        choices=("reference", "zero"),
        default="reference",
        help="Whether the fixed damping term tracks commanded velocity or damps measured velocity toward zero.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DesignConfig:
    return DesignConfig(
        base_period=float(args.base_period),
        cycles=int(args.cycles),
        harmonic_count=int(args.harmonics),
        sample_rate=float(args.sample_rate),
        constraint_rate=float(args.constraint_rate),
        amplitude_scale=float(args.amplitude_scale),
        max_joint_velocity=float(args.max_joint_velocity),
        max_joint_acceleration=float(args.max_joint_acceleration),
        max_joint_jerk=float(args.max_joint_jerk),
        natural_frequency_min_hz=float(args.natural_frequency_min_hz),
        natural_frequency_max_hz=float(args.natural_frequency_max_hz),
        damping_ratio=float(args.damping_ratio),
        bandwidth_samples=int(args.bandwidth_samples),
        candidate_count=int(args.candidates),
        seed=int(args.seed),
        correlation_penalty=float(args.correlation_penalty),
        damping_target=str(args.damping_target),
        repeat_cycles=not bool(args.nonrepeating),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = config_from_args(args)
    started = time.perf_counter()
    try:
        result = design_trajectory(config)
        manifest = write_outputs(Path(args.output_dir), config, result, time.perf_counter() - started)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    information = [metrics["predicted_information"] for metrics in manifest["joint_metrics"].values()]
    print(f"Wrote stiffness-optimized trajectory to {Path(args.output_dir).resolve()}")
    print(f"Design completed in {manifest['elapsed_sec']:.3f} s")
    print(f"Minimum per-joint predicted information: {min(information):.6e}")
    print(f"Peak velocity: {manifest['verification']['peak_velocity']:.3f} rad/s")
    print(f"Peak acceleration: {manifest['verification']['peak_acceleration']:.3f} rad/s^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
