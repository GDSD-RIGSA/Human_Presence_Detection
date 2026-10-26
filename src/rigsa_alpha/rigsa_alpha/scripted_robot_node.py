"""Scripted B2 for machines without unitree_mujoco: patrols the hallway with a played-back trot gait.

Publishes the same topics as b2_bridge (so everything downstream is identical):
  /tf  map -> base_footprint -> base_link,  /joint_states,  /operator/status
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from rigsa_alpha import scenario
from rigsa_alpha.common import make_tf, spin, yaw_quat
from rigsa_alpha.mujoco_world import JOINTS

RATE_HZ = 30.0
GAIT_HZ = 1.8
BODY_Z = 0.58
STAND_THIGH, STAND_CALF = 0.67, -1.3      # Unitree's B2 standing pose


class ScriptedRobot(Node):
    def __init__(self):
        super().__init__('rigsa_scripted_robot')
        self.tf = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.status_pub = self.create_publisher(String, '/operator/status', 10)
        self.t0 = time.monotonic()
        self.create_timer(1.0 / RATE_HZ, self.step)
        self.create_timer(1.0, lambda: self.status_pub.publish(String(data=json.dumps(
            dict(robot='scripted B2 (patrol)')))))
        self.get_logger().info('Scripted B2 patrolling the hallway')

    def step(self):
        t = time.monotonic() - self.t0
        x, y, yaw = scenario.robot_pose(t)
        phase = 2 * math.pi * GAIT_HZ * t
        now = self.get_clock().now().to_msg()
        self.tf.sendTransform([
            make_tf(now, 'map', 'base_footprint', (x, y, 0.0), yaw_quat(yaw)),
            make_tf(now, 'base_footprint', 'base_link', (0.0, 0.0, BODY_Z + 0.012 * math.sin(2 * phase))),
        ])
        q = []
        for leg in ('FR', 'FL', 'RR', 'RL'):
            ph = phase + (0.0 if leg in ('FL', 'RR') else math.pi)   # trot: diagonal pairs together
            lift = max(0.0, math.cos(ph))
            q += [0.0, STAND_THIGH + 0.25 * math.sin(ph) - 0.15 * lift, STAND_CALF - 0.35 * lift]
        js = JointState()
        js.header.stamp = now
        js.name, js.position = JOINTS, q
        self.joint_pub.publish(js)


def main():
    rclpy.init()
    spin(ScriptedRobot())


if __name__ == '__main__':
    main()
