"""Replay an offline SysID trajectory in MuJoCo without ROS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install MuJoCo before running this script: pip install mujoco") from exc
    return mujoco


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _load_trajectory(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    joint_names = list(payload.get("joint_names", []))
    points = list(payload.get("points", []))
    if not joint_names:
        raise ValueError(f"{source}: missing joint_names")
    if len(points) < 2:
        raise ValueError(f"{source}: expected at least two trajectory points")
    times = np.asarray([float(point["time"]) for point in points], dtype=np.float64)
    q = np.asarray([point["positions"] for point in points], dtype=np.float64)
    dq = np.asarray([point.get("velocities", [0.0] * len(joint_names)) for point in points], dtype=np.float64)
    ddq = np.asarray([point.get("accelerations", [0.0] * len(joint_names)) for point in points], dtype=np.float64)
    tau = np.asarray([point.get("torque_preview", [0.0] * len(joint_names)) for point in points], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{source}: trajectory times must be strictly increasing")
    return {"path": str(source), "joint_names": joint_names, "times": times, "q": q, "dq": dq, "ddq": ddq, "tau": tau}


def _interp(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    out = np.zeros(values.shape[1], dtype=np.float64)
    for idx in range(values.shape[1]):
        out[idx] = np.interp(t, times, values[:, idx])
    return out


def _joint_addresses(mujoco, model, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    qpos_adr = []
    dof_adr = []
    joint_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint named {name!r}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Joint {name!r} must be a hinge joint for this replay script")
        qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        dof_adr.append(int(model.jnt_dofadr[joint_id]))
        joint_ids.append(int(joint_id))
    return np.asarray(qpos_adr, dtype=np.int64), np.asarray(dof_adr, dtype=np.int64), joint_ids


def _actuator_for_joints(mujoco, model, joint_ids: list[int]) -> list[int | None]:
    actuators: list[int | None] = []
    for joint_id in joint_ids:
        actuator_id = None
        for idx in range(model.nu):
            trn_id = int(model.actuator_trnid[idx, 0])
            trn_obj = int(model.actuator_trntype[idx])
            if trn_obj == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id == joint_id:
                actuator_id = idx
                break
        actuators.append(actuator_id)
    return actuators


def _set_initial_state(mujoco, model, data, qpos_adr: np.ndarray, dof_adr: np.ndarray, q0: np.ndarray, dq0: np.ndarray) -> None:
    data.qpos[qpos_adr] = q0
    data.qvel[dof_adr] = dq0
    mujoco.mj_forward(model, data)


def simulate(args: argparse.Namespace) -> dict:
    mujoco = _import_mujoco()
    trajectory = _load_trajectory(args.trajectory_json)
    model = mujoco.MjModel.from_xml_path(str(Path(args.model_path).expanduser().resolve()))
    data = mujoco.MjData(model)
    qpos_adr, dof_adr, joint_ids = _joint_addresses(mujoco, model, trajectory["joint_names"])
    actuators = _actuator_for_joints(mujoco, model, joint_ids)

    model.opt.timestep = float(args.timestep)
    times = trajectory["times"]
    q_ref = trajectory["q"]
    dq_ref = trajectory["dq"]
    ddq_ref = trajectory["ddq"]
    tau_ff = trajectory["tau"] if args.use_feedforward else np.zeros_like(trajectory["tau"])

    duration = float(times[-1])
    if args.duration > 0.0:
        duration = min(duration, float(args.duration))
    steps = max(1, int(math.ceil(duration / model.opt.timestep)) + 1)
    log_every = max(1, int(round(float(args.log_dt) / model.opt.timestep)))

    _set_initial_state(mujoco, model, data, qpos_adr, dof_adr, q_ref[0], dq_ref[0])

    log_t = []
    log_q = []
    log_dq = []
    log_q_des = []
    log_dq_des = []
    log_tau_cmd = []
    log_tau_ff = []

    use_actuators = args.control_mode == "actuator"
    if use_actuators and any(actuator is None for actuator in actuators):
        missing = [name for name, actuator in zip(trajectory["joint_names"], actuators) if actuator is None]
        raise ValueError(f"control-mode actuator requested, but no joint actuator was found for: {missing}")

    for step in range(steps):
        t = min(float(data.time), duration)
        q_des = _interp(times, q_ref, t)
        dq_des = _interp(times, dq_ref, t)
        tau_des = _interp(times, tau_ff, t)
        q = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
        dq = np.asarray(data.qvel[dof_adr], dtype=np.float64)
        tau_cmd = tau_des + float(args.kp) * (q_des - q) + float(args.kd) * (dq_des - dq)
        if args.torque_limit > 0.0:
            tau_cmd = np.clip(tau_cmd, -float(args.torque_limit), float(args.torque_limit))

        if use_actuators:
            data.ctrl[:] = 0.0
            for joint_i, actuator_id in enumerate(actuators):
                data.ctrl[int(actuator_id)] = tau_cmd[joint_i]
        else:
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = tau_cmd

        if step % log_every == 0 or step == steps - 1:
            log_t.append(t)
            log_q.append(q.copy())
            log_dq.append(dq.copy())
            log_q_des.append(q_des.copy())
            log_dq_des.append(dq_des.copy())
            log_tau_cmd.append(tau_cmd.copy())
            log_tau_ff.append(tau_des.copy())

        if t >= duration:
            break
        mujoco.mj_step(model, data)

    return {
        "trajectory": trajectory,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "control_mode": args.control_mode,
        "times": np.asarray(log_t),
        "q": np.vstack(log_q),
        "dq": np.vstack(log_dq),
        "q_ref": np.vstack(log_q_des),
        "dq_ref": np.vstack(log_dq_des),
        "tau_cmd": np.vstack(log_tau_cmd),
        "tau_ff": np.vstack(log_tau_ff),
    }


def write_outputs(output_dir: Path, args: argparse.Namespace, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = result["trajectory"]["joint_names"]
    times = result["times"]
    q = result["q"]
    dq = result["dq"]
    q_ref = result["q_ref"]
    dq_ref = result["dq_ref"]
    tau_cmd = result["tau_cmd"]
    tau_ff = result["tau_ff"]
    q_err = q_ref - q
    dq_err = dq_ref - dq

    np.savez_compressed(
        output_dir / "mujoco_replay.npz",
        times=times,
        q=q,
        dq=dq,
        q_ref=q_ref,
        dq_ref=dq_ref,
        q_error=q_err,
        dq_error=dq_err,
        tau_cmd=tau_cmd,
        tau_ff=tau_ff,
        joint_names=np.asarray(joint_names),
    )

    header = (
        ["time"]
        + [f"{name}.q" for name in joint_names]
        + [f"{name}.q_ref" for name in joint_names]
        + [f"{name}.q_error" for name in joint_names]
        + [f"{name}.dq" for name in joint_names]
        + [f"{name}.dq_ref" for name in joint_names]
        + [f"{name}.dq_error" for name in joint_names]
        + [f"{name}.tau_cmd" for name in joint_names]
    )
    with (output_dir / "mujoco_replay.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in zip(times, q, q_ref, q_err, dq, dq_ref, dq_err, tau_cmd):
            writer.writerow([float(row[0]), *row[1].tolist(), *row[2].tolist(), *row[3].tolist(), *row[4].tolist(), *row[5].tolist(), *row[6].tolist(), *row[7].tolist()])

    rms_q = np.sqrt(np.mean(q_err**2, axis=0))
    max_q = np.max(np.abs(q_err), axis=0)
    rms_dq = np.sqrt(np.mean(dq_err**2, axis=0))
    manifest = {
        "schema": "franka_sysid_mujoco_replay_v1",
        "created_wall_time": time.time(),
        "model_path": result["model_path"],
        "trajectory_json": result["trajectory"]["path"],
        "control_mode": result["control_mode"],
        "kp": args.kp,
        "kd": args.kd,
        "use_feedforward": args.use_feedforward,
        "torque_limit": args.torque_limit,
        "timestep": args.timestep,
        "joint_names": joint_names,
        "tracking": {
            "rms_position_error_rad": rms_q.tolist(),
            "max_position_error_rad": max_q.tolist(),
            "rms_velocity_error_rad_s": rms_dq.tolist(),
        },
        "files": {
            "npz": "mujoco_replay.npz",
            "csv": "mujoco_replay.csv",
        },
    }
    (output_dir / "mujoco_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.no_plots:
        write_plots(output_dir, times, q, q_ref, q_err, tau_cmd, joint_names)


def write_plots(output_dir: Path, times: np.ndarray, q: np.ndarray, q_ref: np.ndarray, q_err: np.ndarray, tau_cmd: np.ndarray, joint_names: list[str]) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(10, 12), sharex=True)
    for idx, axis in enumerate(axes):
        axis.plot(times, q_ref[:, idx], label="ref", linewidth=1.2)
        axis.plot(times, q[:, idx], label="sim", linewidth=1.0)
        axis.set_ylabel(joint_names[idx])
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "position_tracking.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(10, 12), sharex=True)
    for idx, axis in enumerate(axes):
        axis.plot(times, q_err[:, idx])
        axis.set_ylabel(joint_names[idx])
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "position_error.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(10, 12), sharex=True)
    for idx, axis in enumerate(axes):
        axis.plot(times, tau_cmd[:, idx])
        axis.set_ylabel(joint_names[idx])
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "commanded_torque.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="MuJoCo MJCF/XML model path. URDF may work if your MuJoCo build supports it.")
    parser.add_argument("--trajectory-json", required=True, help="trajectory.json from franka_sysid_optimize_v3_offline.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--control-mode", choices=["qfrc", "actuator"], default="qfrc")
    parser.add_argument("--kp", type=float, default=450.0)
    parser.add_argument("--kd", type=float, default=45.0)
    parser.add_argument("--use-feedforward", action="store_true")
    parser.add_argument("--torque-limit", type=float, default=87.0)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--log-dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=0.0, help="Optional max replay duration in seconds.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"franka_sysid_mujoco_replay_{stamp}").expanduser().resolve()
    result = simulate(args)
    write_outputs(output_dir, args, result)
    q_err = result["q_ref"] - result["q"]
    print(f"Wrote MuJoCo replay package: {output_dir}")
    print(f"RMS position error rad: {np.sqrt(np.mean(q_err**2, axis=0)).tolist()}")
    print(f"Max position error rad: {np.max(np.abs(q_err), axis=0).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
