#!/bin/bash
# Records one scenario bag and beeps timed cues on the laptop speaker (1 beep = walk, 2 = stop, 3 = done).
# Usage: record_with_cues.sh <bag name> <duration s> "<t>:<beeps>" ...   (cue times in seconds from the start)
# Writes <bag name>.cues.txt with the actual cue times (seconds after recording started) for labelling.
cd "$(dirname "$0")/../bags"
source ../install/setup.bash > /dev/null   # colcon build in the repo root first
name=$1; dur=$2; shift 2
ros2 bag record -o "$name" /radar/points /tf /tf_static > "$name.rec.log" 2>&1 &
rec=$!
sleep 1.5                       # let the recorder subscribe
t0=$(date +%s.%N)
: > "$name.cues.txt"
for cue in "$@"; do
  at=${cue%%:*}; text=${cue#*:}
  now=$(echo "$(date +%s.%N) - $t0" | bc)
  wait_s=$(echo "$at - $now" | bc)
  [ "$(echo "$wait_s > 0" | bc)" = 1 ] && sleep "$wait_s"
  printf '%.1f\t%s\n' "$(echo "$(date +%s.%N) - $t0 + 1.5" | bc)" "$text" >> "$name.cues.txt"
  for ((b = 0; b < text; b++)); do paplay /usr/share/sounds/freedesktop/stereo/bell.oga; sleep 0.15; done &
done
now=$(echo "$(date +%s.%N) - $t0" | bc)
rest=$(echo "$dur - $now" | bc)
[ "$(echo "$rest > 0" | bc)" = 1 ] && sleep "$rest"
kill -INT $rec; wait $rec 2>/dev/null
ros2 bag info "$name" 2>/dev/null | grep -E 'Duration|radar/points'
cat "$name.cues.txt"
