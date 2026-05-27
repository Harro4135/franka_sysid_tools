# Franka SysID Tools

ROS 2 helper package for collecting Franka/Panda free-space telemetry for Isaac Sim Robot Setup System Identification.

The package provides one executable:

```bash
ros2 run franka_sysid_tools franka_sysid_collect --execute --output-dir ~/sysid_runs/franka_001
```

It plans joint-space excitation motions with MoveIt 2, publishes an aligned `control_msgs/msg/JointTrajectoryControllerState` telemetry stream on `/sysid/controller_state`, records a ROS 2 bag, and writes a topic-map YAML that the Isaac SysID importer can load.

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

You need a working Franka MoveIt 2 setup first: robot bringup, controllers, `/joint_states`, and MoveIt planning/execution.

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

## Common Options

```bash
--group panda_arm
--joint-states-topic /joint_states
--telemetry-topic /sysid/controller_state
--cycles 2
--samples-per-cycle 6
--amplitude-scale 0.75
--velocity-scale 0.25
--acceleration-scale 0.25
--storage mcap
--no-record-bag
```

Start conservatively on real hardware. Keep `--velocity-scale`, `--acceleration-scale`, and `--amplitude-scale` low until the trajectory is known to be safe for the cell.
