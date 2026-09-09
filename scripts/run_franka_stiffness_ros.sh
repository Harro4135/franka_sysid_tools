#!/bin/bash
# Build and run the stiffness-only Franka v3 experiment on the ROS 2 machine.
#
# From a checkout located at <ros-workspace>/src/franka_sysid_tools:
#   bash scripts/run_franka_stiffness_ros.sh
#
# No-motion plan generation and collision preflight only:
#   bash scripts/run_franka_stiffness_ros.sh --preflight-only
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [ -n "${FRANKA_ROS_WS:-}" ]; then
  ROS_WS="$FRANKA_ROS_WS"
elif [ "$(basename -- "$(dirname -- "$REPO_ROOT")")" = "src" ]; then
  ROS_WS="$(cd -- "$REPO_ROOT/../.." && pwd)"
else
  echo "FATAL: cannot infer the ROS workspace from $REPO_ROOT"
  echo "Set FRANKA_ROS_WS to the workspace containing this repository."
  exit 2
fi

EXECUTE=1
BUILD=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --preflight-only) EXECUTE=0 ;;
    --skip-build) BUILD=0 ;;
    -h|--help)
      echo "Usage: bash scripts/run_franka_stiffness_ros.sh [--preflight-only] [--skip-build]"
      echo "Environment: FRANKA_ROS_WS, SESSION, OUTROOT, FOLLOW_ACTION, SETTLE_SEC"
      exit 0
      ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
  shift
done

if [ ! -f "$REPO_ROOT/package.xml" ]; then
  echo "FATAL: package.xml is missing from $REPO_ROOT"
  exit 2
fi

if [ "$BUILD" = "1" ]; then
  echo "== Building franka_sysid_tools in $ROS_WS"
  cd "$ROS_WS"
  colcon build --packages-select franka_sysid_tools --symlink-install
fi

if [ ! -f "$ROS_WS/install/setup.bash" ]; then
  echo "FATAL: missing $ROS_WS/install/setup.bash; build the workspace first"
  exit 2
fi
# shellcheck disable=SC1091
source "$ROS_WS/install/setup.bash"

SESSION="${SESSION:-franka_stiffness_$(date +%Y%m%d_%H%M%S)}"
OUTROOT="${OUTROOT:-$HOME/sysid_runs/$SESSION}"
SETTLE_SEC="${SETTLE_SEC:-0}"
export SESSION OUTROOT SETTLE_SEC

echo "== ROS stiffness run"
echo "   repository: $REPO_ROOT"
echo "   session: $SESSION"
echo "   output: $OUTROOT"

cd "$REPO_ROOT"
if [ "$EXECUTE" = "1" ]; then
  exec bash "$SCRIPT_DIR/franka_stiffness_capture.sh" --execute
else
  exec bash "$SCRIPT_DIR/franka_stiffness_capture.sh"
fi
