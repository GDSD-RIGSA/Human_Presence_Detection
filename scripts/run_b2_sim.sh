#!/bin/bash
# RIG-47: start the B2 in unitree_mujoco (stock scene.xml: B2 on a flat floor) and bridge it to ROS 2 (b2_sim.launch.py).
#   scripts/run_b2_sim.sh [name:=value ...]      extra launch arguments, e.g. stand:=false
# Folders can be overridden in the environment. Ctrl+C stops everything.
: "${B2_RADAR_WS:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this repo = the colcon workspace (colcon build here)
: "${UNITREE_MUJOCO:=$HOME/unitree_mujoco}"      # built unitree_mujoco (simulate/build/unitree_mujoco)
: "${UNITREE_ROS2:=$HOME/unitree_ros2}"          # unitree_ros2 (setup_local.sh)
: "${B2_WS:=$HOME/b2_ws}"                        # b2_description + b2_sim_control
: "${LOG_DIR:=$B2_RADAR_WS/log}"
die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "$LOG_DIR"

source /opt/ros/humble/setup.bash
source "$UNITREE_ROS2/setup_local.sh" >/dev/null || die "$UNITREE_ROS2/setup_local.sh missing"   # Cyclone on lo
export ROS_DOMAIN_ID=1                       # unitree_mujoco -i 1
# lo has no multicast, so Cyclone discovers peers by unicast over participant indices (default max 9)
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General>
  <Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>40</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
source "$B2_WS/install/setup.bash" || die "$B2_WS/install/setup.bash missing (b2_description, b2_sim_control)"
source "$B2_RADAR_WS/install/setup.bash" || die "$B2_RADAR_WS/install/setup.bash missing: build b2_radar first"

pgrep -af '[b]2_sim_control' && die "another b2_sim_control node is running: it would fight the stand node for the B2's motors"
SIM_BIN=$UNITREE_MUJOCO/simulate/build/unitree_mujoco
[ -x "$SIM_BIN" ] || die "$SIM_BIN not found (build unitree_mujoco)"

# unitree_mujoco must use its own libddsc (unitree_sdk2), not ROS's: with ROS on LD_LIBRARY_PATH it aborts in dds_write
env -u LD_LIBRARY_PATH "$SIM_BIN" -r b2 -s scene.xml -i 1 -n lo > "$LOG_DIR/unitree_mujoco.log" 2>&1 &
SIM=$!
trap 'kill $SIM 2>/dev/null' EXIT
sleep 3
kill -0 $SIM 2>/dev/null || { tail "$LOG_DIR/unitree_mujoco.log"; die "unitree_mujoco failed"; }

ros2 launch b2_radar b2_sim.launch.py "$@"
