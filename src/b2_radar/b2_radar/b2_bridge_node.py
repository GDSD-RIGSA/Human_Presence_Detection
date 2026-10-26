"""Bridge from the Unitree B2 (unitree_mujoco, or later the real robot) to standard ROS 2 topics.

Reads the Unitree DDS topics that unitree_mujoco publishes exactly like the real B2:
  /sportmodestate  unitree_go/SportModeState  body position (and velocity) in the simulator's world frame
  /lowstate        unitree_go/LowState        IMU orientation and the 12 motor angles (FR, FL, RR, RL x hip, thigh, calf)
Publishes:
  /tf              map -> base_footprint -> base_link (the B2 body, root of the b2_description URDF)
  /joint_states    sensor_msgs/JointState with the URDF joint names, for robot_state_publisher
  /operator/status std_msgs/String (JSON): robot name and body height, 1 Hz

The simulator's world origin is placed at `post` in the map (x, y, yaw); (0, 0, 0) by default, so map = sim world.
Needs the unitree_go messages: source ~/unitree_ros2/setup_local.sh (or setup.sh for the real robot) first.
"""
import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from b2_radar.common import JOINTS, make_tf, spin, yaw_quat

IMU_IN_BASE = np.array([0.0, -0.02341, 0.04927])   # imu site in base_link (unitree_robots/b2/b2.xml)


def quat_wxyz_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class B2Bridge(Node):
    def __init__(self):
        super().__init__('b2_bridge')
        try:
            from unitree_go.msg import LowState, SportModeState
        except ImportError:
            raise SystemExit('unitree_go messages not found. Run "source ~/unitree_ros2/setup_local.sh" '
                             '(and this repo\'s install/setup.bash) in this terminal first.')
        self.declare_parameter('post', [0.0, 0.0, 0.0])
        self.declare_parameter('robot_name', 'unitree_mujoco B2')
        self.post = [float(v) for v in self.get_parameter('post').value]
        self.pos = None
        self.quat = None       # w, x, y, z
        self.q = None
        self.last = 0.0
        self.create_subscription(SportModeState, '/sportmodestate', self.on_sport, qos_profile_sensor_data)
        self.create_subscription(LowState, '/lowstate', self.on_low, qos_profile_sensor_data)
        self.tf = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.status_pub = self.create_publisher(String, '/operator/status', 10)
        self.create_timer(1.0 / 50.0, self.publish)
        self.create_timer(1.0, self.publish_status)
        self.create_timer(5.0, self.check)
        self.get_logger().info('Waiting for /sportmodestate and /lowstate from the B2 (ROS_DOMAIN_ID=1 for unitree_mujoco)')

    def on_sport(self, msg):
        self.pos = np.array(msg.position, dtype=float)
        self.last = time.monotonic()

    def on_low(self, msg):
        self.quat = [float(v) for v in msg.imu_state.quaternion]
        self.q = [float(msg.motor_state[i].q) for i in range(12)]

    def check(self):
        if self.pos is None:
            self.get_logger().warn('No /sportmodestate yet. Is unitree_mujoco running, and is this terminal on '
                                   'CycloneDDS with ROS_DOMAIN_ID=1?')

    def publish(self):
        if self.pos is None or self.quat is None:
            return
        now = self.get_clock().now().to_msg()
        R = quat_wxyz_to_matrix(*self.quat)
        base = self.pos - R @ IMU_IN_BASE                 # sensor gives the imu site; we want base_link
        px, py, pyaw = self.post                          # simulator world -> map
        c, s = math.cos(pyaw), math.sin(pyaw)
        mx, my = px + c * base[0] - s * base[1], py + s * base[0] + c * base[1]
        w, x, y, z = self.quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # base_footprint: on the floor under the body, heading only; base_link adds height, roll and pitch
        fx, fy, fz, fw = yaw_quat(yaw)
        rel = _quat_mul((fw, -fx, -fy, -fz), (w, x, y, z))   # footprint^-1 * body = roll and pitch only
        self.tf.sendTransform([
            make_tf(now, 'map', 'base_footprint', (mx, my, 0.0), yaw_quat(yaw + pyaw)),
            make_tf(now, 'base_footprint', 'base_link', (0.0, 0.0, base[2]), (rel[1], rel[2], rel[3], rel[0])),
        ])
        if self.q is not None:
            js = JointState()
            js.header.stamp = now
            js.name = JOINTS
            js.position = self.q
            self.joint_pub.publish(js)

    def publish_status(self):
        alive = time.monotonic() - self.last < 2.0
        self.status_pub.publish(String(data=json.dumps(dict(
            robot=self.get_parameter('robot_name').value if alive else 'no data',
            height_m=None if self.pos is None else round(float(self.pos[2]), 2)))))


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def main():
    rclpy.init()
    spin(B2Bridge())


if __name__ == '__main__':
    main()
