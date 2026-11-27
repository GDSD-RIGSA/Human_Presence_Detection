#!/bin/bash
# Records a real-radar test for the accuracy evaluation into ~/rigsa_bags/eval/<name>, with a labels.csv to fill in.
# Start the real radar first (scripts/run_room_radar.sh), then:
#     scripts/record_eval_bag.sh <name> [seconds]        e.g.  scripts/record_eval_bag.sh two_crossing 60
# Time 0 of the labels is when "Recording" is printed.
NAME=${1:?usage: record_eval_bag.sh <name> [seconds]}
SECS=${2:-60}
source "$(dirname "$0")/../tools/rigsa_env.sh"
DIR=~/rigsa_bags/eval
mkdir -p "$DIR"
if [ -e "$DIR/$NAME" ]; then echo "$DIR/$NAME already exists"; exit 1; fi
echo "Recording $SECS s of /radar/points to $DIR/$NAME"
timeout -s INT "$SECS" ros2 bag record -o "$DIR/$NAME" /radar/points
cat > "$DIR/$NAME/labels.csv" <<LABELS
start_s,end_s,people,note
0,$SECS,0,EDIT ME: one row per interval, seconds from the start of the recording
LABELS
ros2 bag info "$DIR/$NAME"
echo "Now edit $DIR/$NAME/labels.csv"
