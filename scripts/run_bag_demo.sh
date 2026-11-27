#!/bin/bash
# DEMO FROM A RECORDING (no radar needed): like run_live_demo.sh, but the radar points come from a recorded
# real-radar bag in bags/, looping.
#   scripts/run_bag_demo.sh [bag name or folder] [name:=value ...]     default s7_four
#   scripts/run_bag_demo.sh s2_one bag_loop:=false
# Ctrl+C stops everything.
source "$(dirname "${BASH_SOURCE[0]}")/rigsa_common.sh"
: "${RIGSA_BAGS:=$RIGSA_REPO/bags}"
BAG=s7_four
if [ -n "$1" ] && [[ "$1" != *:=* ]]; then BAG=$1; shift; fi
[ -d "$BAG" ] || BAG=$RIGSA_BAGS/$BAG
[ -f "$BAG/metadata.yaml" ] || die "no rosbag at $BAG (recordings: $(cd "$RIGSA_BAGS" && ls -d */ 2>/dev/null | tr -d / | tr "\n" " "))"
start_mujoco
launch_alpha bag_demo.log robot:=unitree view:=3d radar:=bag bag:="$(readlink -f "$BAG")" \
    detector_params:="$DETECTOR_PARAMS" "$@"
