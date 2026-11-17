#!/bin/bash
# FULL SIMULATION: the B2 in unitree_mujoco with a SIMULATED radar and simulated people walking the building;
# the evaluator scores the detections against the ground truth.
#   scripts/run_sim_demo.sh [name:=value ...]          e.g.  radar_gui:=true  seed:=3
# Ctrl+C stops everything.
source "$(dirname "${BASH_SOURCE[0]}")/rigsa_common.sh"
start_mujoco
launch_alpha sim_demo.log robot:=unitree "$@"
