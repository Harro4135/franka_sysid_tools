import json

import numpy as np

from franka_sysid_tools.franka_sysid_optimize_stiffness_offline import (
    DesignConfig,
    FRANKA_AMPLITUDES,
    FRANKA_LIMITS,
    build_argument_parser,
    design_trajectory,
    stiffness_sensitivity_gain_squared,
    write_outputs,
)


def test_production_defaults_use_boosted_motion_envelope():
    np.testing.assert_allclose(FRANKA_AMPLITUDES, [0.40, 0.28, 0.38, 0.25, 0.38, 0.25, 0.38])
    args = build_argument_parser().parse_args(["--output-dir", "unused"])
    assert args.max_joint_acceleration == 2.0


def _config(**overrides):
    values = {
        "base_period": 5.0,
        "cycles": 2,
        "harmonic_count": 5,
        "sample_rate": 40.0,
        "constraint_rate": 200.0,
        "amplitude_scale": 0.8,
        "max_joint_velocity": 0.75,
        "max_joint_acceleration": 1.5,
        "max_joint_jerk": 0.0,
        "natural_frequency_min_hz": 0.4,
        "natural_frequency_max_hz": 2.0,
        "damping_ratio": 0.7,
        "bandwidth_samples": 7,
        "candidate_count": 24,
        "seed": 1234,
        "correlation_penalty": 0.2,
        "damping_target": "reference",
    }
    values.update(overrides)
    return DesignConfig(**values)


def test_log_stiffness_sensitivity_is_zero_at_dc_and_informative_near_bandwidth():
    gains = stiffness_sensitivity_gain_squared(
        np.asarray([1e-6, 1.0]),
        np.asarray([1.0]),
        damping_ratio=0.7,
    )

    assert gains.shape == (2, 1)
    assert gains[0, 0] < 1e-20
    assert gains[1, 0] > 0.1


def test_designed_trajectory_is_reproducible_feasible_and_stops_at_cycle_boundaries():
    config = _config()
    first = design_trajectory(config)
    second = design_trajectory(config)

    np.testing.assert_array_equal(first["positions"], second["positions"])
    q = np.asarray(first["positions"])
    dq = np.asarray(first["velocities"])
    ddq = np.asarray(first["accelerations"])
    assert np.all(q >= FRANKA_LIMITS[:, 0] - 1e-12)
    assert np.all(q <= FRANKA_LIMITS[:, 1] + 1e-12)
    assert np.max(np.abs(dq)) < config.max_joint_velocity
    assert np.max(np.abs(ddq)) < config.max_joint_acceleration

    samples_per_cycle = int(config.base_period * config.sample_rate)
    boundary_indices = np.arange(config.cycles + 1) * samples_per_cycle
    assert np.max(np.abs(dq[boundary_indices])) < 1e-10
    assert np.max(np.abs(ddq[boundary_indices])) < 1e-10


def test_more_candidates_never_reduce_first_joint_information():
    baseline = design_trajectory(_config(candidate_count=1))
    optimized = design_trajectory(_config(candidate_count=32))

    baseline_design = baseline["joint_designs"][0]
    optimized_design = optimized["joint_designs"][0]
    assert optimized_design.predicted_information >= baseline_design.predicted_information


def test_independent_seed_changes_trajectory_and_outputs_collector_schema(tmp_path):
    config = _config(candidate_count=12)
    result = design_trajectory(config)
    independent = design_trajectory(_config(candidate_count=12, seed=9876))

    assert not np.array_equal(result["positions"], independent["positions"])
    manifest = write_outputs(tmp_path, config, result, elapsed_sec=0.01)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "success"
    assert (tmp_path / "_SUCCESS").read_text(encoding="utf-8").strip() == "success"
    assert trajectory["schema"] == "franka_sysid_offline_fourier_trajectory_v1"
    assert trajectory["points"][0]["time"] == 0.0
    assert len(trajectory["points"][0]["positions"]) == 7
    assert manifest["model"]["identified_parameter"] == "per-joint drive stiffness K"
