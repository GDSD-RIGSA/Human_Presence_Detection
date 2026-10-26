"""Recorded radar: replays /radar/points from a rosbag2 folder as if the radar were live.

Unlike `ros2 bag play`, it publishes only the radar points (the room-test bags also hold a /tf_static that would
clash with the robot's TF), restamps every cloud with the current time and keeps the original frame spacing, so the
detector sees a steady clock even when the bag loops.

Publishes:  /radar/points  sensor_msgs/PointCloud2 in radar_link (as recorded)
Parameters: bag (rosbag2 folder), loop (default true), rate (playback speed, default 1.0)
"""
import os

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from b2_radar.common import RADAR_FRAME, spin


def read_bag(path):
    """[(t in s, PointCloud2)] of /radar/points, in recording order."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/radar/points']))
    out = []
    while reader.has_next():
        _, data, t = reader.read_next()
        out.append((t * 1e-9, deserialize_message(data, PointCloud2)))
    return out


class BagRadar(Node):
    def __init__(self):
        super().__init__('bag_radar')
        path = os.path.expanduser(self.declare_parameter('bag', '').value)
        self.loop = self.declare_parameter('loop', True).value
        self.rate = float(self.declare_parameter('rate', 1.0).value)
        if not os.path.isdir(path):
            raise SystemExit('bag_radar: bag folder not found: %r' % path)
        self.msgs = read_bag(path)
        if not self.msgs:
            raise SystemExit('bag_radar: no /radar/points in %s' % path)
        self.name = os.path.basename(path.rstrip('/'))
        self.pub = self.create_publisher(PointCloud2, '/radar/points', 10)   # reliable, like radar_driver
        self.get_logger().info('Replaying %d radar frames (%.1f s) from %s%s' % (
            len(self.msgs), self.msgs[-1][0] - self.msgs[0][0], path, ', looping' if self.loop else ''))
        self.i = 0
        self.start_wall = self.now()
        self.passes = 1
        self.create_timer(0.01, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        t0 = self.msgs[0][0]
        while self.i < len(self.msgs) and (self.msgs[self.i][0] - t0) / self.rate <= self.now() - self.start_wall:
            msg = self.msgs[self.i][1]
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = msg.header.frame_id or RADAR_FRAME
            self.pub.publish(msg)
            self.i += 1
        if self.i >= len(self.msgs):
            if not self.loop:
                self.get_logger().info('Bag %s finished' % self.name, once=True)
                return
            self.i = 0
            self.start_wall = self.now() + 0.1   # one empty frame gap between passes
            self.passes += 1
            self.get_logger().info('Bag %s: pass %d' % (self.name, self.passes))


def main():
    rclpy.init()
    spin(BagRadar())


if __name__ == '__main__':
    main()
