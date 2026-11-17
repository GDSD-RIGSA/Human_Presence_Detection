#!/bin/bash
# ROOM TEST: the real TI radar on a stand, no robot (robot:=none), improved detector, RViz + operator panel.
#   scripts/run_room_radar.sh [name:=value ...]        e.g.  radar_height:=1.2     RADAR_CFG=profile.cfg scripts/run_room_radar.sh
# Before: plug the radar in (/dev/ttyACM0 + ttyACM1), close the TI visualizer. Ctrl+C stops it.
source "$(dirname "${BASH_SOURCE[0]}")/rigsa_common.sh"
check_radar_ports
launch_alpha room_radar.log robot:=none radar_height:=1.0 radar_cfg:="$RADAR_CFG" \
    cli_port:="$RADAR_CLI" data_port:="$RADAR_DATA" detector_params:="$DETECTOR_PARAMS" "$@"
