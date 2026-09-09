#!/bin/bash
# Higher-energy Franka v3 D-optimal capture with a genuinely held-out validation plan.
#
# Usage on the ROS 2 robot machine:
#   URDF=/path/to/panda_fixed_base.urdf bash scripts/franka_v3_high_energy_capture.sh
#   URDF=/path/to/panda_fixed_base.urdf bash scripts/franka_v3_high_energy_capture.sh --execute
set -euo pipefail

SESSION="${SESSION:-franka_v3_high_energy_001}"
OUTROOT="${OUTROOT:-$HOME/sysid_runs/$SESSION}"
URDF="${URDF:?set URDF=/path/to/panda_fixed_base.urdf}"
FOLLOW_ACTION="${FOLLOW_ACTION:-/panda_arm_controller/follow_joint_trajectory}"
JOINT_STATES_TOPIC="${JOINT_STATES_TOPIC:-/joint_states}"
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/franka_robot_state_broadcaster/robot_state}"

# This envelope raises the older v3 defaults (0.70 amplitude, 0.65 rad/s,
# 1.50 rad/s^2) only to the already-used v2 velocity/acceleration envelope.
SAMPLE_RATE_HZ="${SAMPLE_RATE_HZ:-100}"
MAX_VELOCITY_RAD_S="${MAX_VELOCITY_RAD_S:-0.85}"
MAX_ACCELERATION_RAD_S2="${MAX_ACCELERATION_RAD_S2:-1.75}"
TRAIN_AMPLITUDE_SCALE="${TRAIN_AMPLITUDE_SCALE:-0.90}"
TRAIN_BASE_PERIOD_SEC="${TRAIN_BASE_PERIOD_SEC:-6.0}"
TRAIN_CYCLES="${TRAIN_CYCLES:-5}"
TRAIN_SEED="${TRAIN_SEED:-20260909}"
VALIDATION_AMPLITUDE_SCALE="${VALIDATION_AMPLITUDE_SCALE:-0.85}"
VALIDATION_BASE_PERIOD_SEC="${VALIDATION_BASE_PERIOD_SEC:-6.7}"
VALIDATION_CYCLES="${VALIDATION_CYCLES:-3}"
VALIDATION_SEED="${VALIDATION_SEED:-20261111}"
MIN_RATE_HZ="${MIN_RATE_HZ:-45}"
TARGET_RATE_HZ="${TARGET_RATE_HZ:-100}"
EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1

PLAN_ROOT="$OUTROOT/offline_plans_high_energy_v1"
TRAIN_PLAN="$PLAN_ROOT/train"
VALIDATION_PLAN="$PLAN_ROOT/validation"
CAPTURE="$OUTROOT/capture"
mkdir -p "$OUTROOT"

echo "== Franka v3 higher-energy session: $SESSION"
echo "   output: $OUTROOT"
echo "   envelope: amplitude=$TRAIN_AMPLITUDE_SCALE, velocity=$MAX_VELOCITY_RAD_S rad/s, acceleration=$MAX_ACCELERATION_RAD_S2 rad/s^2"

if ! ros2 action list 2>/dev/null | grep -q "$FOLLOW_ACTION"; then
  echo "FATAL: FollowJointTrajectory action '$FOLLOW_ACTION' not found"; exit 2
fi
if ! ros2 topic list 2>/dev/null | grep -q "^$ROBOT_STATE_TOPIC$"; then
  echo "FATAL: $ROBOT_STATE_TOPIC is required for measured link-side tau_J"; exit 2
fi
if ! ros2 run franka_sysid_tools franka_sysid_collect_v3 --help 2>&1 \
     | grep -q -- "--validation-trajectory-json"; then
  echo "FATAL: installed franka_sysid_collect_v3 is stale; rebuild and source the workspace"; exit 2
fi

RATE=$( { timeout 15 ros2 topic hz --window 200 "$JOINT_STATES_TOPIC" 2>/dev/null || true; } \
        | grep -oE "average rate: [0-9.]+" | tail -n1 | grep -oE "[0-9.]+" || echo 0)
echo "   $JOINT_STATES_TOPIC measured rate: ${RATE:-0} Hz (floor $MIN_RATE_HZ, target >= $TARGET_RATE_HZ)"
python3 -c 'import sys; rate, floor, target = map(float, sys.argv[1:]); print(f"WARN: raise telemetry toward {target:.0f} Hz" if floor <= rate < target else ""); raise SystemExit(0 if rate >= floor else f"joint-state rate {rate:.1f} Hz is below the {floor:.0f} Hz floor")' "$RATE" "$MIN_RATE_HZ" "$TARGET_RATE_HZ"

if [ ! -f "$TRAIN_PLAN/trajectory.json" ] || [ ! -f "$TRAIN_PLAN/manifest.json" ]; then
  echo "== Optimizing higher-energy training plan"
  python3 -m franka_sysid_tools.franka_sysid_optimize_v3_offline \
    --urdf-path "$URDF" \
    --output-dir "$TRAIN_PLAN" \
    --sample-rate "$SAMPLE_RATE_HZ" \
    --base-period "$TRAIN_BASE_PERIOD_SEC" \
    --cycles "$TRAIN_CYCLES" \
    --amplitude-scale "$TRAIN_AMPLITUDE_SCALE" \
    --seed "$TRAIN_SEED" \
    --max-joint-velocity "$MAX_VELOCITY_RAD_S" \
    --max-joint-acceleration "$MAX_ACCELERATION_RAD_S2" \
    --ipopt-max-iter 500
fi

if [ ! -f "$VALIDATION_PLAN/trajectory.json" ] || [ ! -f "$VALIDATION_PLAN/manifest.json" ]; then
  echo "== Optimizing independent validation plan"
  python3 -m franka_sysid_tools.franka_sysid_optimize_v3_offline \
    --urdf-path "$URDF" \
    --output-dir "$VALIDATION_PLAN" \
    --sample-rate "$SAMPLE_RATE_HZ" \
    --base-period "$VALIDATION_BASE_PERIOD_SEC" \
    --cycles "$VALIDATION_CYCLES" \
    --amplitude-scale "$VALIDATION_AMPLITUDE_SCALE" \
    --seed "$VALIDATION_SEED" \
    --max-joint-velocity "$MAX_VELOCITY_RAD_S" \
    --max-joint-acceleration "$MAX_ACCELERATION_RAD_S2" \
    --ipopt-max-iter 500
fi

echo "== Collision preflight for train and validation (no motion)"
PREFLIGHT="$OUTROOT/preflight_$(date +%Y%m%d_%H%M%S)"
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --preflight-only \
  --require-validation \
  --trajectory-json "$TRAIN_PLAN/trajectory.json" \
  --validation-trajectory-json "$VALIDATION_PLAN/trajectory.json" \
  --follow-action "$FOLLOW_ACTION" \
  --torque-source franka-robot-state \
  --robot-state-topic "$ROBOT_STATE_TOPIC" \
  --include-effort \
  --sample-rate "$SAMPLE_RATE_HZ" \
  --max-joint-velocity "$MAX_VELOCITY_RAD_S" \
  --max-joint-acceleration "$MAX_ACCELERATION_RAD_S2" \
  --collision-check-stride 2 \
  --output-dir "$PREFLIGHT"

if [ "$EXECUTE" != "1" ]; then
  echo "== Plans and no-motion collision preflight complete"
  echo "   Inspect $TRAIN_PLAN/{positions,velocities,accelerations,torque_preview}.png"
  echo "   Inspect $VALIDATION_PLAN/{positions,velocities,accelerations,torque_preview}.png"
  echo "   Re-run with --execute to collect the hardware bag."
  exit 0
fi

if [ -e "$CAPTURE" ]; then
  echo "FATAL: capture output already exists: $CAPTURE (choose a new SESSION)"; exit 2
fi
read -r -p "Robot clear, cell safe, E-stop in hand. Execute higher-energy v3 train + holdout? [yes/NO] " CONFIRM
[ "$CONFIRM" = "yes" ] || { echo "aborted"; exit 1; }

echo "== Capturing higher-energy training followed by excluded validation"
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --execute \
  --require-validation \
  --trajectory-json "$TRAIN_PLAN/trajectory.json" \
  --validation-trajectory-json "$VALIDATION_PLAN/trajectory.json" \
  --follow-action "$FOLLOW_ACTION" \
  --torque-source franka-robot-state \
  --robot-state-topic "$ROBOT_STATE_TOPIC" \
  --include-effort \
  --sample-rate "$SAMPLE_RATE_HZ" \
  --max-joint-velocity "$MAX_VELOCITY_RAD_S" \
  --max-joint-acceleration "$MAX_ACCELERATION_RAD_S2" \
  --collision-check-stride 2 \
  --output-dir "$CAPTURE"

python3 "$(dirname "$0")/verify_v3_holdout.py" "$CAPTURE"
ros2 bag info "$CAPTURE/bag"
echo "== Complete: $CAPTURE"
