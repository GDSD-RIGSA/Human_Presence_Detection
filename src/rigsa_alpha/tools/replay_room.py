#!/usr/bin/env python3
"""Offline replay of real-radar room recordings (robot:=none) through the detector, without ROS launch.

    python3 tools/replay_room.py <bag> [<bag> ...] [--params room_improved.yaml] [--points]

Reads /radar/points from each rosbag2 folder, puts the points in the room-mode map frame (radar at the map origin,
radar_height above the floor, facing +x: the same static TF site_node publishes) and feeds them to Detector with
the room-mode tuning from alpha.launch.py, optionally overridden by a params yaml. Prints alerts and IDs per bag.
--points prints per-frame point statistics (intensity, velocity, range) instead, for tuning the filters.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rigsa_alpha.common import read_cloud_full   # noqa: E402
from rigsa_alpha.detection import BackgroundMask, Detector   # noqa: E402

ROOM_TRACKING = {'gate': 2.0, 'lost_after': 4.0, 'reacquire_s': 15.0, 'reacquire_m': 3.0, 'eps': 0.6}


def room_mask(res=0.05):
    """The all-free grid site_node publishes in room mode."""
    w, h = int(14.0 / res), int(16.0 / res)
    return BackgroundMask(np.zeros((h, w), dtype=np.int8), res, (-2.0, -8.0))


def read_bag(path):
    """Yields (t, (N, 5) points in radar_link) for every /radar/points message."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/radar/points']))
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        yield msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, read_cloud_full(msg)


def load_params(path):
    """Flat ros__parameters dict from a ROS 2 params yaml."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    out = {}
    for node in doc.values():
        out.update(node.get('ros__parameters', {}))
    return out


def replay(path, params, radar_height=1.0):
    det = Detector(room_mask(), **params)
    alerts, ids, t0 = [], set(), None
    for t, pts in read_bag(path):
        t0 = t if t0 is None else t0
        pts = pts.copy()
        pts[:, 2] += radar_height
        for tr in det.update(pts, t, sensor_xy=(0.0, 0.0)):
            alerts.append((t - t0, tr.id, float(np.hypot(*tr.pos)), tr.n_points))
        ids.update(tr.id for tr in det.confirmed_tracks())
    return alerts, ids


def point_stats(path):
    rows = [pts for _, pts in read_bag(path)]
    allp = np.vstack(rows) if rows else np.zeros((0, 5))
    rng = np.hypot(allp[:, 0], allp[:, 1])
    print('%s: %d frames, %d points (%.1f / frame)' % (os.path.basename(path), len(rows), len(allp), len(allp) / max(len(rows), 1)))
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10)):
        sel = (rng >= lo) & (rng < hi)
        if sel.any():
            i, v = allp[sel, 3], np.abs(allp[sel, 4])
            print('  %d-%d m: %5d pts  intensity dB p10/p50/p90 %.1f/%.1f/%.1f  |v| p50/p90 %.2f/%.2f  zero-v %.0f%%'
                  % (lo, hi, sel.sum(), *np.percentile(i, [10, 50, 90]), *np.percentile(v, [50, 90]), 100 * (v < 1e-3).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--params', help='yaml with detector overrides (on top of the room tuning)')
    ap.add_argument('--points', action='store_true')
    a = ap.parse_args()
    params = dict(ROOM_TRACKING)
    if a.params:
        params.update(load_params(a.params))
    for bag in a.bags:
        if a.points:
            point_stats(bag)
            continue
        alerts, ids = replay(bag, params)
        print('%s: %d alerts, %d IDs' % (os.path.basename(bag.rstrip('/')), len(alerts), len(ids)))
        for t, tid, dist, n in alerts:
            print('   %5.1f s  %-4s %4.1f m  %2d pts' % (t, tid, dist, n))


if __name__ == '__main__':
    main()
