# Stage-2 capture session — plan and checklist

One-command driver: [`scripts/stage2_capture_session.sh`](scripts/stage2_capture_session.sh)
(preflight + offline plan + dry-run by default; add `--execute` for hardware).

## Why this suite (sim-sim benchmark evidence, simsim_v2, 2026-07-07)

| Goal | Evidence | Stage-2 answer |
|---|---|---|
| Drive gains | Multisine + inverse_dynamics leaves stiffness/damping 18-40% off at the noise floor (R1); raw-PD structure recovers them to ~2% when excitation is adequate (R3a) | D-optimal Fourier suite (v3), fit in the matched controller structure |
| Friction | 3/7 joints fail even under raw PD on the multisine — weakly excited joints lack velocity reversals (R3a) | v2 per-joint friction sweeps at 0.12/0.28/0.50 rad/s peaks, 2 repeats |
| Mass / CoM | Multisine leaves ~2/3 of a mass deviation below the noise floor even with gains frozen (R3b: 34.5% recovery); CoM unidentified in every run | Static holds + slow segments with **link-side tau_J** (`--torque-source franka-robot-state`), solved via load-side regression, not the rollout torque channel |
| Logging rate | Halving 30 Hz -> 15 Hz biases every joint-level family 40-56% low (R5) | Preflight hard-fails < 25 Hz; raise bringup `/joint_states` toward 100+ Hz |
| Command delay | franka_v2_003 measured ~527 ms transport delay; uncorrected it confounds damping/friction | After import, read the quality report's lag estimate and set `telemetry.command_delay_seconds` before solving |

## Pre-session (once)

- [ ] Rebuild the package on the ROS machine (`colcon build --packages-select franka_sysid_tools`) — v3 gained `--torque-source` / `--robot-state-topic` (this session's edit).
- [ ] Fixed-base Panda URDF available and matching the 7 controlled joints (`URDF=...`).
- [ ] `franka_robot_state_broadcaster` running (tau_J source).
- [ ] Raise the bringup `/joint_states` publish rate as high as the stack allows; verify with `ros2 topic hz`.
- [ ] Offline plan solved and previews inspected (`offline_plan/torque_preview.png` — effort scale must sit inside Franka limits with margin).

## Session order

1. Dry-run (script default): plan validation + collision preflight, no motion.
2. Optional: replay `trajectory.json` in MoveIt fake hardware or MuJoCo (`franka_sysid_sim_mujoco.py`) if this is the first execution of a new plan.
3. `--execute`: v3 D-optimal suite, then v2 friction sweeps + static holds.
4. QC on the spot: `ros2 bag info` durations match manifests; effort channel non-empty; no controller aborts in `phase_events.jsonl`.

## Safety

- Defaults are conservative (`amplitude-scale 0.70`, vel 0.65 rad/s, acc 1.5 rad/s²); collision preflight stays ON.
- The D-optimal suite is more aggressive than the old multisine by design. First hardware run: keep defaults; only raise amplitude after reviewing tracking error and torque margins from the first pass.
- E-stop in hand during all `--execute` phases.

## Post-session (Isaac Sim machine)

- [ ] Import each bag with its topic map; confirm `torque_semantics: link_side`.
- [ ] Telemetry-quality report: common-mode lag -> `telemetry.command_delay_seconds`; per-joint residual spread noted (actuator latency, do not fold in).
- [ ] Solve joint families (gains + friction) on v3 train phases, validate on `d_optimal_validation`.
- [ ] Solve mass/CoM by load-side regression on static holds; cross-check against the rollout solve's mass.
- [ ] Run the sim-sim harness against the new spec before trusting the solve (`tools/headless_sysid_simsim.py`) — recovering known truth is the only loud failure mode.
