#!/bin/bash
# RIGSA terminal setup. Source it in every terminal that runs the demo (instead of setup_local.sh):
#     source tools/rigsa_env.sh            (from the repo root, after colcon build)
# Simulation only: loopback interface and ROS_DOMAIN_ID=1, like unitree_mujoco. Never use it for the real B2.
# Folders can be overridden in the environment: UNITREE_ROS2, B2_WS.

RIGSA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this repo = the colcon workspace
: "${UNITREE_ROS2:=$HOME/unitree_ros2}"
: "${B2_WS:=$HOME/b2_ws}"

if [ -f "$UNITREE_ROS2/setup_local.sh" ]; then
  source "$UNITREE_ROS2/setup_local.sh" > /dev/null    # Humble + unitree_go messages + CycloneDDS on lo
  # Same as setup_local.sh, plus a higher participant limit: on lo (no multicast) CycloneDDS allows only about
  # 10 ROS processes per domain by default, and the demo starts more ("rmw_create_node: failed to create domain").
  export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>100</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
else
  source /opt/ros/humble/setup.bash                    # no unitree_ros2: robot:=scripted, replay and none still work
fi
export ROS_DOMAIN_ID=1
[ -f "$B2_WS/install/setup.bash" ] && source "$B2_WS/install/setup.bash"          # b2_description, b2_sim_control
[ -f "$RIGSA_REPO/install/setup.bash" ] && source "$RIGSA_REPO/install/setup.bash" # b2_radar, rigsa_alpha
echo "RIGSA env: RMW=${RMW_IMPLEMENTATION:-default} ROS_DOMAIN_ID=$ROS_DOMAIN_ID workspace=$RIGSA_REPO"
