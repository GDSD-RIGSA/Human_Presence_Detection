"""Radar simulator: the mmWave radar on the B2, plus the people walking in the building.

Neither unitree_mujoco nor Gazebo can simulate an mmWave radar or move people around, so this node keeps its
own MuJoCo world (building, people, and a copy of the B2 that follows the real simulation) and casts the
radar's rays in it. It works with any robot source that publishes TF and /joint_states (b2_bridge, scripted_robot).

Subscribes: /tf (map -> radar_link, map -> base_link), /joint_states
Publishes:  /radar/points        sensor_msgs/PointCloud2 in radar_link: x, y, z, intensity (SNR dB), velocity
            /sim/people          std_msgs/String (JSON): true positions + visible flag (for the evaluator only)
            /sim/people_markers  visualization_msgs/MarkerArray: true positions, drawn faintly in RViz
Parameters: gui (own MuJoCo window with building, people and B2, default false), b2_xml (Unitree's b2.xml for the
            full B2 meshes in that window), seed.
"""
import contextlib
import json
import math
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rigsa_alpha import scenario
from rigsa_alpha.common import RADAR_FRAME, make_cloud, spin, transform_matrix
from rigsa_alpha.floor import FURNITURE, WALLS
from rigsa_alpha.mujoco_world import RADAR_HALF_AZ, RADAR_HZ, RADAR_RANGE, World


class RadarSim(Node):
    def __init__(self):
        super().__init__('rigsa_radar_sim')
        self.declare_parameter('gui', False)
        self.declare_parameter('b2_xml', '')
        self.declare_parameter('seed', -1)
        seed = self.get_parameter('seed').value
        b2_xml = self.get_parameter('b2_xml').value or None
        self.world = World(WALLS, FURNITURE, len(scenario.PEOPLE), b2_xml=b2_xml, seed=None if seed < 0 else seed)
        if b2_xml and not self.world.full_b2:
            self.get_logger().warn('b2_xml %s not found; showing a simplified B2' % b2_xml)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joints = {}
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.points_pub = self.create_publisher(PointCloud2, '/radar/points', 10)
        self.truth_pub = self.create_publisher(String, '/sim/people', 10)
        self.truth_markers = self.create_publisher(MarkerArray, '/sim/people_markers', 10)
        self.viewer = None
        if self.get_parameter('gui').value:
            self.open_viewer()
        self.t0 = time.monotonic()
        self.prev = None
        self.last_origin = None
        self.warned = False
        self.create_timer(1.0 / RADAR_HZ, self.step)
        self.get_logger().info('Radar simulator running: %d people, radar %.0f Hz, %.0f m, +-%.0f deg'
                               % (len(scenario.PEOPLE), RADAR_HZ, RADAR_RANGE, RADAR_HALF_AZ))

    def open_viewer(self):
        try:
            import mujoco
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.world.model, self.world.data,
                                                       show_left_ui=False, show_right_ui=False)
            with self.viewer.lock():   # camera follows the B2 from behind and above; drag with the mouse to change
                cam = self.viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.world.b2_body
                cam.distance, cam.elevation, cam.azimuth = 11.0, -50.0, 20.0
        except Exception as e:   # no display, missing GLFW, ...
            self.get_logger().warn('MuJoCo viewer not available (%s)' % e)
            self.viewer = None

    def on_joints(self, msg):
        self.joints = dict(zip(msg.name, msg.position))

    def lookup(self, frame, stamp=False):
        try:
            tf = self.tf_buffer.lookup_transform('map', frame, Time(), timeout=Duration(seconds=0.0))
            return (transform_matrix(tf.transform), tf.header.stamp) if stamp else transform_matrix(tf.transform)
        except TransformException as e:
            if not self.warned:
                self.get_logger().warn('Waiting for the robot pose (TF map -> %s): %s' % (frame, e))
                self.warned = True
            return (None, None) if stamp else None

    def people(self, t, dt):
        people = scenario.people_poses(t)
        vel = np.zeros((len(people), 3))
        if self.prev is not None:
            for i, (a, b) in enumerate(zip(people, self.prev)):
                if a is not None and b is not None:
                    vel[i, :2] = [(a[0] - b[0]) / dt, (a[1] - b[1]) / dt]
        self.prev = people
        return people, vel

    def step(self):
        T_base, (T_radar, pose_stamp) = self.lookup('base_link'), self.lookup(RADAR_FRAME, stamp=True)
        if T_base is None or T_radar is None:
            return
        t = time.monotonic() - self.t0
        people, vel = self.people(t, 1.0 / RADAR_HZ)
        origin, rot = T_radar[:3, 3], T_radar[:3, :3]
        sensor_vel = np.zeros(3)
        if self.last_origin is not None:
            sensor_vel = (origin - self.last_origin) * RADAR_HZ
        self.last_origin = origin.copy()
        with self.world_lock():   # the viewer draws from another thread
            self.world.set_people(people, vel)
            self.set_robot(T_base)
            self.world.forward()
            points = self.world.radar_scan(origin, rot, sensor_vel)
        now = self.get_clock().now()
        # stamped with the time of the robot pose used for the scan, so the detector places the points with exactly
        # that pose (with `now` it would interpolate TF to a slightly different one, which matters when turning)
        self.points_pub.publish(make_cloud(points, pose_stamp))
        self.publish_truth(people, origin, rot, now)
        self.draw()

    def world_lock(self):
        return self.viewer.lock() if self.viewer is not None else contextlib.nullcontext()

    def set_robot(self, T):
        R = T[:3, :3]
        w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        quat = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
        self.world.set_robot(T[:3, 3], quat, self.joints)

    def draw(self):
        """Shows the world just scanned in the viewer window (10 Hz; drawing more often slowed the radar on CPU-only
        machines, because the viewer holds the lock while it renders)."""
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            return
        self.viewer.sync()

    def publish_truth(self, people, origin, rot, now):
        out, arr = [], MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, p in enumerate(people):
            if p is None:
                continue
            local = rot.T @ (np.array([p[0], p[1], 1.0]) - origin)
            rng, az = math.hypot(local[0], local[1]), math.degrees(math.atan2(local[1], local[0]))
            visible = (rng < RADAR_RANGE and abs(az) < RADAR_HALF_AZ
                       and self.world.line_of_sight(origin, (p[0], p[1], 1.0)))
            out.append(dict(person=i, x=round(p[0], 2), y=round(p[1], 2), visible=bool(visible)))
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'sim_people', i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p[0], p[1], 0.85
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.45
            m.scale.z = 1.7
            m.color = ColorRGBA(r=0.3, g=0.5, b=1.0, a=0.25)
            arr.markers.append(m)
        self.truth_pub.publish(String(data=json.dumps(dict(stamp=now.nanoseconds * 1e-9, people=out))))
        self.truth_markers.publish(arr)


def main():
    rclpy.init()
    node = RadarSim()

    def close_viewer():
        if node.viewer is not None:
            node.viewer.close()
            time.sleep(0.3)   # let the viewer thread finish before MuJoCo data is freed

    spin(node, close_viewer)
    if node.viewer is not None:
        import os
        os._exit(0)           # skip interpreter teardown: the viewer's GL thread can crash it


if __name__ == '__main__':
    main()
