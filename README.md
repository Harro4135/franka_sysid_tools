# Franka SysID Tools

ROS 2 helper package for collecting Franka/Panda free-space telemetry for Isaac Sim Robot Setup System Identification.

The package provides two executables:

```bash
ros2 run franka_sysid_tools franka_sysid_collect --execute --output-dir ~/sysid_runs/franka_001
```

It plans joint-space excitation motions through the standard MoveIt 2 `MoveGroup` action, executes the resulting trajectory through `ExecuteTrajectory`, publishes an aligned `control_msgs/msg/JointTrajectoryControllerState` telemetry stream on `/sysid/controller_state`, records a ROS 2 bag, and writes a topic-map YAML that the Isaac SysID importer can load.

For more methodical SysID data collection, use the v2 executable:

```bash
ros2 run franka_sysid_tools franka_sysid_collect_v2 --execute --output-dir ~/sysid_runs/franka_v2_001
```

V2 uses MoveIt to reposition to the start of each phase, then commands explicit designed joint trajectories through a `FollowJointTrajectory` controller action. This makes the recorded `reference.positions` the actual designed excitation rather than a MoveIt-retimed waypoint path. Before execution, V2 samples the direct trajectory waypoints through MoveIt's `/check_state_validity` service so self/world collisions are caught before the robot moves.

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

Leave `--skip-moveit-start` off for normal use so MoveIt can reposition between phases. Only use it when the robot is already in the safe free-space envelope and the direct joint trajectory controller is known to be configured correctly.

Collision checking is enabled for `--execute` runs by default. It checks every Nth generated waypoint, plus the final point, where N is `--collision-check-stride`. At the default `--sample-rate 50` and `--collision-check-stride 10`, that is one checked state about every 0.2 s. This is a sampled state-validity preflight, not a continuous swept-volume proof between samples. If your MoveIt setup exposes the state-validity service under a different name, pass `--collision-check-service`. If you intentionally need to bypass this preflight, pass `--no-collision-check`.
