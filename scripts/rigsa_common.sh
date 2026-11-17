# Shared settings for the scripts/run_*.sh launchers (sourced, not run). Override any variable in
# <repo>/local.env or in the environment, e.g.
#     RADAR_CLI=/dev/ttyACM2 RADAR_DATA=/dev/ttyACM3 scripts/run_room_radar.sh
RIGSA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$RIGSA_REPO/local.env" ] && source "$RIGSA_REPO/local.env"

: "${UNITREE_MUJOCO:=$HOME/unitree_mujoco}"      # built unitree_mujoco (simulate/build/unitree_mujoco)
: "${RADAR_CLI:=/dev/ttyACM0}"                   # TI radar CLI port (USB)
: "${RADAR_DATA:=/dev/ttyACM1}"                  # TI radar data port (USB)
: "${RADAR_CFG:=profile_test.cfg}"               # bare name = shipped in the package (radar_cfg/); or a path
: "${DETECTOR_PARAMS=room_improved}"             # bare name = package params/; or a path; set empty = previous detector
: "${OPERATOR:=$USER}"
: "${LOG_DIR:=$RIGSA_REPO/log}"

die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "$LOG_DIR"

# ROS 2 Humble + unitree messages + CycloneDDS on lo, ROS_DOMAIN_ID=1, b2_ws and this repo (simulation network only)
source "$RIGSA_REPO/tools/rigsa_env.sh" || die "$RIGSA_REPO/tools/rigsa_env.sh missing"
RIGSA_SHARE="$(ros2 pkg prefix rigsa_alpha 2>/dev/null)/share/rigsa_alpha"
[ -d "$RIGSA_SHARE" ] || die "rigsa_alpha not installed: run colcon build in $RIGSA_REPO"
[[ "$RADAR_CFG" == */* ]] || RADAR_CFG=$RIGSA_SHARE/radar_cfg/$RADAR_CFG
[ -z "$DETECTOR_PARAMS" ] || [[ "$DETECTOR_PARAMS" == */* ]] || DETECTOR_PARAMS=$RIGSA_SHARE/params/$DETECTOR_PARAMS.yaml

check_radar_ports() {
    for t in "$RADAR_CLI" "$RADAR_DATA"; do
        [ -e "$(readlink -f "$t")" ] || die "radar port $t missing (USB unplugged? see ls /dev/ttyACM*)"
        [ -r "$t" ] && [ -w "$t" ] || die "no read/write access to $t (dialout group / TI udev rules)"
    done
    if command -v lsof >/dev/null && lsof "$(readlink -f "$RADAR_CLI")" "$(readlink -f "$RADAR_DATA")" >/dev/null 2>&1; then
        die "the radar ports are in use (TI mmWave Demo Visualizer or another launch?): lsof $RADAR_CLI $RADAR_DATA"
    fi
}

# Ctrl+C: ros2 launch stops its nodes, then the script exits and cleanup stops unitree_mujoco.
SIM=
cleanup() {
    trap - EXIT INT TERM
    [ -n "$SIM" ] || return
    kill "$SIM" 2>/dev/null
    for _ in 1 2 3 4 5 6; do kill -0 "$SIM" 2>/dev/null || return; sleep 0.5; done
    kill -9 "$SIM" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# The B2 in unitree_mujoco (scene with the RIGSA building), in the background; log in $LOG_DIR/unitree_mujoco.log
start_mujoco() {
    pgrep -x unitree_mujoco >/dev/null && die "unitree_mujoco is already running (pkill unitree_mujoco)"
    pgrep -af 'lib/b2_sim_control/' && die "another b2_sim_control node is running: it would fight the stand node for the B2's motors"
    local bin=$UNITREE_MUJOCO/simulate/build/unitree_mujoco scene=$UNITREE_MUJOCO/unitree_robots/b2/scene_rigsa.xml
    [ -x "$bin" ] || die "$bin not found (build unitree_mujoco)"
    [ -f "$scene" ] || ros2 run rigsa_alpha make_unitree_scene "$scene" || die "could not write $scene"
    # unitree_mujoco must use its own libddsc (unitree_sdk2), not ROS's: with ROS on LD_LIBRARY_PATH it aborts in dds_write
    env -u LD_LIBRARY_PATH "$bin" -r b2 -s scene_rigsa.xml -i 1 -n lo > "$LOG_DIR/unitree_mujoco.log" 2>&1 &
    SIM=$!
    sleep 3
    kill -0 "$SIM" 2>/dev/null || { tail "$LOG_DIR/unitree_mujoco.log"; die "unitree_mujoco failed"; }
}

# ros2 launch rigsa_alpha alpha.launch.py <args>, output also in $LOG_DIR/<log name>
launch_alpha() {
    local log=$1; shift
    ros2 launch rigsa_alpha alpha.launch.py operator:="$OPERATOR" "$@" 2>&1 | tee -i "$LOG_DIR/$log"
}
