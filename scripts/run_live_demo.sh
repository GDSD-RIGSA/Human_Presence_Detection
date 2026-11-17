#!/bin/bash
# LIVE DEMO: the simulated B2 (unitree_mujoco) stands guard; the radar on its nose is the REAL TI AWR1443 (USB).
# People the radar sees appear in a 3D RViz view next to the B2 and in the operator panel; alerts go to PostgreSQL.
#   scripts/run_live_demo.sh [name:=value ...]        e.g.  RADAR_CFG=profile.cfg scripts/run_live_demo.sh
# Before: plug the radar in (/dev/ttyACM0 + ttyACM1), close the TI visualizer, SOP2 jumper off. Ctrl+C stops everything.
source "$(dirname "${BASH_SOURCE[0]}")/rigsa_common.sh"
check_radar_ports
start_mujoco
launch_alpha live_demo.log robot:=unitree view:=3d radar:=real radar_cfg:="$RADAR_CFG" \
    cli_port:="$RADAR_CLI" data_port:="$RADAR_DATA" detector_params:="$DETECTOR_PARAMS" "$@"
