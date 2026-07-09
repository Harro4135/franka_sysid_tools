#!/bin/bash
# Stage-2 Franka SysID capture session (run on the ROS 2 machine).
#
# Informed by the sim-sim ground-truth benchmark (isaacsim.robot_setup.sysid,
# franka_v2_003/sysid_session_2026-07-06/simsim_v2):
#   - The coupled multisine cannot pin gains/CoM/inertia (noise-floor limited);
#     the regressor-optimized (D-optimal) Fourier suite is the excitation that
#     recovers them in sim-sim.  -> v3 D-optimal phases are the core capture.
#   - Friction only identifies with strong per-joint velocity reversals.
#     -> v2 friction sweeps at multiple peak velocities.
#   - Mass/CoM need the torque channel (load-side regression), which requires
#     link-side tau_J. -> --torque-source franka-robot-state on both collectors.
#   - 15 Hz logging destroys joint-level identifiability (R5: 40-56% errors);
#     30 Hz is the working floor. -> preflight refuses to execute below
#     MIN_RATE_HZ; raise the bringup /joint_states rate as high as possible.
#
# Usage:
#   ./stage2_capture_session.sh            # preflight + offline plan + dry-run only
#   ./stage2_capture_session.sh --execute  # additionally run on the robot
set -euo pipefail

# ----------------------------- configuration --------------------------------
SESSION="${SESSION:-franka_stage2_001}"
OUTROOT="${OUTROOT:-$HOME/sysid_runs/$SESSION}"
URDF="${URDF:?set URDF=/path/to/panda_fixed_base.urdf}"
FOLLOW_ACTION="${FOLLOW_ACTION:-/panda_arm_controller/follow_joint_trajectory}"
JOINT_STATES_TOPIC="${JOINT_STATES_TOPIC:-/joint_states}"
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/franka_robot_state_broadcaster/robot_state}"
MIN_RATE_HZ=25          # hard floor: sim-sim R5 shows 15 Hz is unusable
TARGET_RATE_HZ=100      # ask bringup for at least this; more is better
EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1

mkdir -p "$OUTROOT"
echo "== Stage-2 session: $SESSION -> $OUTROOT (execute=$EXECUTE)"

# ------------------------------- preflight ----------------------------------
echo "== Preflight"
if ! ros2 action list 2>/dev/null | grep -q "$FOLLOW_ACTION"; then
  echo "FATAL: FollowJointTrajectory action '$FOLLOW_ACTION' not found (ros2 action list)"; exit 2
fi
if ! ros2 topic list 2>/dev/null | grep -q "^$ROBOT_STATE_TOPIC$"; then
  echo "WARN: $ROBOT_STATE_TOPIC not published - tau_J (link-side torque) will be missing."
  echo "      Start the franka_robot_state_broadcaster or Stage-2 mass/CoM goals are compromised."
  [ "$EXECUTE" = "1" ] && exit 2
fi

RATE=$(timeout 15 ros2 topic hz --window 200 "$JOINT_STATES_TOPIC" 2>/dev/null \
        | grep -oE "average rate: [0-9.]+" | tail -1 | grep -oE "[0-9.]+" || echo 0)
echo "   $JOINT_STATES_TOPIC measured rate: ${RATE:-0} Hz (floor $MIN_RATE_HZ, target >= $TARGET_RATE_HZ)"
python3 - "$RATE" "$MIN_RATE_HZ" "$TARGET_RATE_HZ" <<'EOF'
import sys
rate, floor, target = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if rate < floor:
    print(f"FATAL: joint_states at {rate:.1f} Hz is below the {floor:.0f} Hz floor "
          "(sim-sim rate ablation: half-rate logging biases every joint-level family 40-56%).")
    sys.exit(2)
if rate < target:
    print(f"WARN: {rate:.1f} Hz works but raise the bringup publish rate toward {target:.0f} Hz "
          "if possible (better command-delay estimation and acceleration content).")
EOF

# --------------------------- offline D-optimal plan --------------------------
PLAN="$OUTROOT/offline_plan"
if [ ! -f "$PLAN/trajectory.json" ]; then
  echo "== Solving offline D-optimal plan (CasADi/IPOPT + Pinocchio)"
  python3 -m franka_sysid_tools.franka_sysid_optimize_v3_offline \
    --urdf-path "$URDF" \
    --output-dir "$PLAN"
  echo "   Inspect $PLAN/torque_preview.png and positions/velocities plots before executing."
else
  echo "== Reusing existing offline plan: $PLAN/trajectory.json"
fi

# ------------------------------- dry runs -----------------------------------
echo "== v3 dry-run (plan validation, no motion)"
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --trajectory-json "$PLAN/trajectory.json" \
  --urdf-path "$URDF" \
  --follow-action "$FOLLOW_ACTION" \
  --torque-source franka-robot-state \
  --robot-state-topic "$ROBOT_STATE_TOPIC" \
  --include-effort \
  --output-dir "$OUTROOT/v3_d_optimal_dryrun"

if [ "$EXECUTE" != "1" ]; then
  echo "== Dry-run complete. Re-run with --execute to capture on hardware."
  exit 0
fi

# ----------------------------- hardware capture ------------------------------
read -r -p "Robot clear, cell safe, E-stop in hand. Execute Stage-2 suite? [yes/NO] " CONFIRM
[ "$CONFIRM" = "yes" ] || { echo "aborted"; exit 1; }

echo "== Capture 1/2: v3 D-optimal suite (warmup, train, fast, static holds, validation)"
ros2 run franka_sysid_tools franka_sysid_collect_v3 \
  --execute \
  --trajectory-json "$PLAN/trajectory.json" \
  --urdf-path "$URDF" \
  --follow-action "$FOLLOW_ACTION" \
  --torque-source franka-robot-state \
  --robot-state-topic "$ROBOT_STATE_TOPIC" \
  --include-effort \
  --output-dir "$OUTROOT/v3_d_optimal"

echo "== Capture 2/2: v2 friction sweeps + static holds (per-joint velocity ladder)"
ros2 run franka_sysid_tools franka_sysid_collect_v2 \
  --execute \
  --follow-action "$FOLLOW_ACTION" \
  --torque-source franka-robot-state \
  --robot-state-topic "$ROBOT_STATE_TOPIC" \
  --include-effort \
  --friction-peak-velocity 0.12 0.28 0.50 \
  --single-joint-repeats 2 \
  --output-dir "$OUTROOT/v2_friction_poses"

# ------------------------------ post-capture QC ------------------------------
echo "== Post-capture QC"
for d in v3_d_optimal v2_friction_poses; do
  echo "-- $d"; ros2 bag info "$OUTROOT/$d/bag" | head -20 || true
done
cat <<'EOF'
Next (on the Isaac Sim machine):
 1. Load each bag with its franka_sysid_topic_map.yaml through the SysID importer.
 2. Check the telemetry-quality report's command-lag estimate; franka_v2_003
    measured ~0.527 s transport delay - set telemetry.command_delay_seconds
    in the run spec BEFORE solving (delay/damping/friction confound otherwise).
 3. Verify torque_semantics: link_side in the topic map (tau_J present).
 4. Solve drives/friction against the v3+v2 data with feedforward matched to
    the executed controller; solve mass/CoM via the load-side regression on
    static holds + slow segments.
EOF
echo "== Session complete: $OUTROOT"
