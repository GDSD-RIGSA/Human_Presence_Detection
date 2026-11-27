#!/bin/bash
# Records the radar data and robot pose from a running demo into a rosbag in ~/rigsa_bags.
# Start a demo first (any robot mode), then: scripts/record_backup_bag.sh [seconds] [name]
# Replay later with:  ros2 launch rigsa_alpha alpha.launch.py robot:=replay   and, in a second terminal,
#                     ros2 bag play ~/rigsa_bags/<name>
SECS=${1:-90}
NAME=${2:-alpha_$(date +%Y%m%d_%H%M)}
source "$(dirname "$0")/../tools/rigsa_env.sh"
mkdir -p ~/rigsa_bags
echo "Recording $SECS s to ~/rigsa_bags/$NAME"
timeout -s INT "$SECS" ros2 bag record -o ~/rigsa_bags/"$NAME" /tf /joint_states /radar/points /sim/people /sim/people_markers
ros2 bag info ~/rigsa_bags/"$NAME"
