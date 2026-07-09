# Franka SysID Tools

ROS 2 helper package for collecting Franka/Panda free-space telemetry for Isaac Sim Robot Setup System Identification.

For a full guided Stage-2 capture (offline D-optimal plan, preflight with a logging-rate gate, v3 + v2 friction/static suites with link-side torque, post-capture QC), see [`STAGE2_SESSION.md`](STAGE2_SESSION.md) and run [`scripts/stage2_capture_session.sh`](scripts/stage2_capture_session.sh). Both v2 and v3 collectors accept `--torque-source franka-robot-state` to publish measured link-side `tau_J` as `feedback.effort` (declared `torque_semantics: link_side` in the topic map) and to record the raw robot-state topic.

The package provides several collection executables:

```bash
ros2 run franka_sysid_tools franka_sysid_collect --execute --output-dir ~/sysid_runs/franka_001
```

It plans joint-space excitation motions through the standard MoveIt 2 `MoveGroup` action, executes the resulting trajectory through `ExecuteTrajectory`, publishes an aligned `control_msgs/msg/JointTrajectoryControllerState` telemetry stream on `/sysid/controller_state`, records a ROS 2 bag, and writes a topic-map YAML that the Isaac SysID importer can load.

For more methodical SysID data collection, use the v2 executable:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v2 --execute --output-dir ~/sysid_runs/franka_v2_001
```

V2 uses MoveIt to reposition to the start of each phase, then commands explicit designed joint trajectories through a `FollowJointTrajectory` controller action. This makes the recorded `reference.positions` the actual designed excitation rather than a MoveIt-retimed waypoint path. Before execution, V2 samples the direct trajectory waypoints through MoveIt's `/check_state_validity` service so self/world collisions are caught before the robot moves.

For an experimental information-dense design, use the v3 D-optimal collector:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --execute \
  --urdf-path /path/to/panda_fixed_base.urdf \
  --output-dir ~/sysid_runs/franka_v3_001
```

V3 uses the same MoveIt start repositioning, direct `FollowJointTrajectory` execution, telemetry publishing, bag recording, and collision preflight as V2. Its coupled excitation phases are finite-Fourier-series trajectories solved as constrained NLPs with CasADi / IPOPT. The D-optimal objective is evaluated on identifiable base columns of Pinocchio's physical Panda torque regressor extracted from the supplied URDF.

There is also an SO-101 variant using the LeRobot arm joint names:

```bash
ros2 run franka_sysid_tools so101_sysid_collect_v2 --execute --output-dir ~/sysid_runs/so101_v2_001
```

The SO-101 script defaults to the five arm joints `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, and `wrist_roll`. It intentionally excludes `gripper` from the excitation suite and uses conservative placeholder limits; verify those against your calibrated URDF or joint-limit file before increasing amplitudes on hardware.

## Layout

```text
franka_sysid_tools/
  package.xml
  setup.py
  setup.cfg
  config/
    franka_sysid_topic_map.yaml
  franka_sysid_tools/
    franka_sysid_collect.py
    franka_sysid_collect_v2.py
    franka_sysid_collect_v3.py
    so101_sysid_collect_v2.py
```

## Build On The ROS Machine

Copy or clone this repo into a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <this-repo-url> franka_sysid_tools

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select franka_sysid_tools
source install/setup.bash
```

You need a working Franka MoveIt 2 setup first: robot bringup, controllers, `/joint_states`, and MoveIt planning/execution. The script does not require `moveit_py`; it uses `moveit_msgs` actions so it works on Humble installs where `ros-humble-moveit-py` is unavailable.

## Dry Run

This plans every segment without executing or recording:

```bash
ros2 run franka_sysid_tools franka_sysid_collect
```

## Collect A Dataset

```bash
ros2 run franka_sysid_tools franka_sysid_collect \
  --execute \
  --output-dir ~/sysid_runs/franka_001
```

Outputs:

```text
~/sysid_runs/franka_001/
  bag/
  franka_sysid_topic_map.yaml
  run_metadata.json
```

Load `bag/` as a ROS 2 bag in the Isaac SysID UI and set `Mapping config` to `franka_sysid_topic_map.yaml`.

## Collect A Methodical V2 Dataset

First find the joint trajectory controller action on the robot:

```bash
ros2 action list | grep follow_joint_trajectory
```

Then run v2 with that action path:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v2 \
  --execute \
  --follow-action /panda_arm_controller/follow_joint_trajectory \
  --output-dir ~/sysid_runs/franka_v2_001
```

V2 writes:

```text
~/sysid_runs/franka_v2_001/
  bag/
  franka_sysid_topic_map.yaml
  run_metadata.json
  collection_manifest.json
  phase_events.jsonl
```

The collection suite is split into phases:

- `warmup_multisine`: low-amplitude controller/friction warmup.
- `friction_sweeps_*`: single-joint sine sweeps at multiple peak velocities.
- `coupled_multisine_train`: coupled multi-joint training excitation.
- `coupled_multisine_fast`: lower-amplitude faster excitation for acceleration terms.
- `static_holds`: steady poses for gravity/bias observations.
- `coupled_multisine_validation`: held-out trajectory for validation, not fitting.

## Collect A D-Optimal V3 Dataset

First find the joint trajectory controller action on the robot:

```bash
ros2 action list | grep follow_joint_trajectory
```

Then run v3 with that action path:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --execute \
  --follow-action /panda_arm_controller/follow_joint_trajectory \
  --urdf-path /path/to/panda_fixed_base.urdf \
  --output-dir ~/sysid_runs/franka_v3_001
```

The v3 suite is split into:

- `warmup_multisine`: low-amplitude controller/friction warmup.
- `d_optimal_train`: physical-regressor D-optimal Fourier NLP training trajectory.
- `d_optimal_fast`: shorter, higher-frequency D-optimal Fourier NLP trajectory for acceleration-dependent terms.
- `static_holds`: steady poses for gravity/bias observations.
- `d_optimal_validation`: held-out D-optimal NLP trajectory from a separate initialization seed.

The manifest records the URDF path, base-regressor rank settings, Fourier harmonics, IPOPT settings, ridge/condition penalty, and per-phase D-optimal scores. The URDF must be a fixed-base Panda model matching the seven controlled arm joints.

## Offline V3 Solve And Sim Replay

You can solve the physical-regressor D-optimal trajectory on a non-ROS machine with only Python, NumPy, CasADi, Pinocchio, and Matplotlib:

```bash
python -m franka_sysid_tools.franka_sysid_optimize_v3_offline \
  --urdf-path /path/to/panda_fixed_base.urdf \
  --output-dir ~/sysid_runs/franka_v3_offline_plan
```

By default the offline optimizer uses seeded random feasible samples to select
base-regressor columns, includes Coulomb/viscous friction columns in the
trajectory scoring objective, and solves a conditioned D-optimal problem. Use
`--objective d_opt` and `--disable-friction-regressor` for the original
inertial-only D-optimal style.

The offline package writes:

```text
~/sysid_runs/franka_v3_offline_plan/
  trajectory.json
  trajectory.csv
  trajectory.npz
  manifest.json
  positions.png
  velocities.png
  accelerations.png
  torque_preview.png
```

The plots and `torque_preview` are a Pinocchio inverse-dynamics preview, useful for checking the shape and effort scale before running through ROS. To replay the offline trajectory in MoveIt fake hardware or another ROS sim, start MoveIt with `use_real_hardware:=false`, then run:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --execute \
  --trajectory-json ~/sysid_runs/franka_v3_offline_plan/trajectory.json \
  --follow-action /panda_arm_controller/follow_joint_trajectory \
  --output-dir ~/sysid_runs/franka_v3_fake_hw_replay
```

When `--trajectory-json` is supplied, the ROS collector skips CasADi/Pinocchio optimization and uses the sampled offline trajectory directly. Collision preflight, MoveIt start repositioning, telemetry publishing, and bag recording still run through the normal v3 execution path.

You can also replay the same offline trajectory in MuJoCo without ROS:

```bash
python -m franka_sysid_tools.franka_sysid_sim_mujoco \
  --model-path /path/to/panda_mujoco.xml \
  --trajectory-json ~/sysid_runs/franka_v3_offline_plan/trajectory.json \
  --use-feedforward \
  --output-dir ~/sysid_runs/franka_v3_mujoco_replay
```

The MuJoCo replay script applies a joint-space PD controller plus optional feedforward torques from the offline Pinocchio preview. By default it writes directly to `qfrc_applied`, which is useful for model-level testing even if the MJCF has no actuators. Use `--control-mode actuator` when your MJCF has torque actuators attached to the Panda joints.

MuJoCo replay writes:

```text
~/sysid_runs/franka_v3_mujoco_replay/
  mujoco_replay.csv
  mujoco_replay.npz
  mujoco_manifest.json
  position_tracking.png
  position_error.png
  commanded_torque.png
```

## Collect An SO-101 V2 Dataset

First find the SO-101 joint trajectory controller action:

```bash
ros2 action list | grep follow_joint_trajectory
```

Then pass that action path if your stack does not use the default:

```bash
ros2 run franka_sysid_tools so101_sysid_collect_v2 \
  --execute \
  --follow-action /so101_arm_controller/follow_joint_trajectory \
  --group so101_arm \
  --output-dir ~/sysid_runs/so101_v2_001
```

If your MoveIt group/controller expects different joint names, update the constants in `franka_sysid_tools/so101_sysid_collect_v2.py` before running. The default excitation is for the SO-101 arm chain only, not the gripper.

## Common Options

```bash
--group panda_arm
--joint-states-topic /joint_states
--telemetry-topic /sysid/controller_state
--move-group-action /move_action
--execute-action /execute_trajectory
--planner-id ''
--pipeline-id ''
--cycles 2
--samples-per-cycle 6
--amplitude-scale 0.75
--velocity-scale 0.25
--acceleration-scale 0.25
--storage mcap
--no-record-bag
```

Start conservatively on real hardware. Keep `--velocity-scale`, `--acceleration-scale`, and `--amplitude-scale` low until the trajectory is known to be safe for the cell.

V2 safety knobs:

```bash
--sample-rate 50
--base-period 8.0
--amplitude-scale 0.70
--friction-peak-velocity 0.12 0.28
--max-joint-velocity 0.65
--max-joint-acceleration 1.50
--collision-check-service /check_state_validity
--collision-check-stride 10
--skip-moveit-start
```

V3 D-optimality knobs:

```bash
--urdf-path /path/to/panda_fixed_base.urdf
--trajectory-json ~/sysid_runs/franka_v3_offline_plan/trajectory.json
--base-regressor-samples 240
--base-regressor-rank-tolerance 1e-8
--fourier-harmonics 5
--d-opt-seed 20260611
--d-opt-score-stride 4
--d-opt-ridge 1e-3
--d-opt-condition-penalty 0.0
--ipopt-max-iter 500
--ipopt-print-level 5
--ipopt-tolerance 1e-6
```

Offline-only rank-aware optimizer knobs:

```bash
--base-regressor-sampling random_feasible
--objective conditioned_d_opt
--condition-penalty 0.05
--include-friction-regressor
--disable-friction-regressor
```

Leave `--skip-moveit-start` off for normal use so MoveIt can reposition between phases. Only use it when the robot is already in the safe free-space envelope and the direct joint trajectory controller is known to be configured correctly.

Collision checking is enabled for `--execute` runs by default. It checks every Nth generated waypoint, plus the final point, where N is `--collision-check-stride`. At the default `--sample-rate 50` and `--collision-check-stride 10`, that is one checked state about every 0.2 s. This is a sampled state-validity preflight, not a continuous swept-volume proof between samples. If your MoveIt setup exposes the state-validity service under a different name, pass `--collision-check-service`. If you intentionally need to bypass this preflight, pass `--no-collision-check`.
