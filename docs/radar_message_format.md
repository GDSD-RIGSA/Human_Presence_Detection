# Radar measurements on ROS 2 (RIG-49)

Two nodes in `rigsa_alpha` publish the same radar message, so everything downstream (the detector, RViz,
rosbag) works the same with the real radar, the simulated radar and a recording:

| Node | Source |
| --- | --- |
| `radar_driver` | The real TI AWR1443BOOST (mmWave SDK 2.x out-of-box demo) over USB |
| `radar_sim` | Ray-cast model of the AWR1443 on the B2's nose, in a MuJoCo copy of the building |
| `b2_radar` `bag_radar` (RIG-48) | A rosbag2 recording of `/radar/points`, replayed and restamped |

## `/radar/points`: `sensor_msgs/msg/PointCloud2`

- **Frame:** `header.frame_id = radar_link`. The frame is x forward (boresight), y left, z up.
  `site_node` publishes the static transform `base_link -> radar_link` at `(0.52, 0.0, 0.06)` m, on the B2's nose.
- **Stamp:** the time the frame was received (driver) or simulated (sim).
- **Layout:** one row (`height = 1`, `width = N` points), little-endian, `is_dense = true`, `point_step = 20` bytes.

| Field | Offset | Type | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `x` | 0 | float32 | m | forward |
| `y` | 4 | float32 | m | left |
| `z` | 8 | float32 | m | up |
| `intensity` | 12 | float32 | dB | Driver: `10 * log10(peak + 1)` from the TI peak value. Sim: SNR in dB. |
| `velocity` | 16 | float32 | m/s | Radial (Doppler) velocity: Doppler bin x velocity resolution |

The field names are defined once, in `rigsa_alpha/common.py` (`CLOUD_FIELDS`), and packed by `make_cloud()`.

- **Rate:** one message per radar frame. That is 10 Hz for the simulator (`RADAR_HZ`) and with the shipped TI
  configs (`frameCfg` period 100 ms).
- **Frames with no detections:** still published, with `width = 0`.
- **QoS:** reliable, depth 10.

### How the driver converts a TI frame

The driver reads the data port (921600 baud) and finds each frame by the magic word `02 01 04 03 06 05 08 07`.
From each frame's detected-points TLV (type 1) it takes the header `numObj, xyzQFormat` and then
`numObj x (range u16, doppler i16, peak u16, x i16, y i16, z i16)`. Each point is converted as follows:

- **Position:** TI's frame is x right, y forward, z up, so the driver maps `radar_link` x = TI y, `radar_link`
  y = -TI x, and `radar_link` z = TI z. Each value is scaled by `1 / 2^xyzQFormat`.
- **Velocity:** `doppler x vel_res`. The resolution `vel_res = lambda / (2 * N_doppler * N_tx * (idle + ramp))`
  is computed from `profileCfg` and `frameCfg` in the `.cfg`.

`test/test_radar_driver.py` builds a synthetic TI frame and checks this conversion, the field layout, the frame
id, empty frames and the velocity resolution.

## `/operator/status`: `std_msgs/msg/String` (JSON)

The driver publishes this once per second, for example `{"robot": "TI AWR1443 (room test, no robot)"}`.

## How to verify

With the real radar (room test) or the simulator running:

```bash
ros2 topic info /radar/points -v          # type sensor_msgs/msg/PointCloud2, publisher radar_driver or radar_sim
ros2 topic hz /radar/points               # ~10 Hz
ros2 topic echo /radar/points --once --field fields   # x, y, z, intensity, velocity (FLOAT32, offsets 0..16)
ros2 topic echo /radar/points --once --field header   # frame_id: radar_link
ros2 run tf2_ros tf2_echo base_link radar_link                 # 0.52 0 0.06
```

The driver also logs `Radar: <n> frames in the last 10 s, <m> points` every 10 s. Expect about 100 frames per
10 s. 0 frames with `CLI "..." -> no reply` on every command means the board is not running the demo firmware;
check that the SOP2 jumper is off and power-cycle the board.

In RViz, add a PointCloud2 display on `/radar/points` with the fixed frame `map` (or `radar_link`) and the
color transformer set to `intensity` or `velocity`.

Offline tests (no radar, no ROS graph):

```bash
source /opt/ros/humble/setup.bash
cd src/rigsa_alpha && python3 -m pytest -q test/test_radar_driver.py
```
