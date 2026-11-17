#!/bin/bash
# LIGHTWEIGHT SIMULATION: no unitree_mujoco; a scripted B2 patrols the hallway with a SIMULATED radar
# and simulated people (evaluator on).
#   scripts/run_scripted_demo.sh [name:=value ...]     e.g.  seed:=3
# Ctrl+C stops it.
source "$(dirname "${BASH_SOURCE[0]}")/rigsa_common.sh"
launch_alpha scripted_demo.log robot:=scripted "$@"
