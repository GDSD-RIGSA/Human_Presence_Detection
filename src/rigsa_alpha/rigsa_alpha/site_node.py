"""Site description: the building map, the radar mount on the B2, and static RViz markers.

Publishes (all latched):
  /map               nav_msgs/OccupancyGrid  building floor (maps/hallway.yaml)
  /tf_static         base_link -> radar_link (where the radar sits on the B2)
  /rigsa/scene       visualization_msgs/MarkerArray  radar field of view, room names, and a box for the robot
                     when no URDF robot model is shown (parameter robot_box)
"""
import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from rigsa_alpha.common import RADAR_FRAME, RADAR_MOUNT, load_map, make_tf, spin
from rigsa_alpha.floor import ROOMS
from rigsa_alpha.mujoco_world import RADAR_HALF_AZ, RADAR_RANGE


class SiteNode(Node):
    def __init__(self):
        super().__init__('rigsa_site')
        share = get_package_share_directory('rigsa_alpha')
        self.declare_parameter('map_yaml', os.path.join(share, 'maps', 'hallway.yaml'))
        self.declare_parameter('robot_box', False)
        self.declare_parameter('room', False)          # room test with a real radar on a stand: no building, no robot
        self.declare_parameter('radar_height', 1.0)    # room test: radar height above the floor, metres
        self.room = self.get_parameter('room').value
        self.radar_height = self.get_parameter('radar_height').value
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.scene_pub = self.create_publisher(MarkerArray, '/rigsa/scene', latched)
        self.static_tf = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()
        tfs = [make_tf(now, 'base_link', RADAR_FRAME, RADAR_MOUNT)]
        if self.room:   # the radar stands at the map origin, facing +x, radar_height above the floor
            tfs.append(make_tf(now, 'map', 'base_link', (-RADAR_MOUNT[0], 0.0, self.radar_height - RADAR_MOUNT[2])))
            self.publish_room()
        else:
            self.publish_map(self.get_parameter('map_yaml').value)
        self.static_tf.sendTransform(tfs)
        self.scene_pub.publish(self.scene_markers(self.get_parameter('robot_box').value))

    def publish_map(self, yaml_path):
        w, h, res, origin, grid = load_map(yaml_path)
        m = OccupancyGrid()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = res
        m.info.width, m.info.height = w, h
        m.info.origin.position.x, m.info.origin.position.y = origin[0], origin[1]
        m.info.origin.orientation.w = 1.0
        m.data = grid.flatten().tolist()
        self.map_pub.publish(m)

    def publish_room(self, res=0.05):
        """All-free grid covering the radar's field of view: nothing is masked as a wall."""
        w, h = int(14.0 / res), int(16.0 / res)
        m = OccupancyGrid()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = res
        m.info.width, m.info.height = w, h
        m.info.origin.position.x, m.info.origin.position.y = -2.0, -8.0
        m.info.origin.orientation.w = 1.0
        m.data = [0] * (w * h)
        self.map_pub.publish(m)

    def marker(self, ns, mid, mtype, frame='map'):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.frame_locked = frame != 'map'
        return m

    def scene_markers(self, robot_box):
        arr = MarkerArray()
        radar = self.marker('robot', 1, Marker.CUBE, RADAR_FRAME)
        radar.scale.x, radar.scale.y, radar.scale.z = 0.08, 0.2, 0.12
        radar.color = ColorRGBA(r=0.0, g=0.9, b=0.46, a=1.0)
        name = self.marker('robot', 2, Marker.TEXT_VIEW_FACING, 'base_link')
        name.text, name.scale.z, name.color = 'radar' if self.room else 'B2', 0.6, ColorRGBA(r=0.85, g=0.4, b=0.0, a=1.0)
        name.pose.position.x, name.pose.position.z = -0.9, 0.6
        arr.markers += [radar, name]
        if robot_box:
            body = self.marker('robot', 0, Marker.CUBE, 'base_link')
            body.scale.x, body.scale.y, body.scale.z = 1.1, 0.45, 0.25
            body.color = ColorRGBA(r=0.95, g=0.55, b=0.1, a=1.0)
            arr.markers.append(body)

        # radar field of view, drawn flat on the floor (the floor is about RADAR_MOUNT z + body height below)
        fov = self.marker('radar_fov', 0, Marker.TRIANGLE_LIST, RADAR_FRAME)
        fov.scale.x = fov.scale.y = fov.scale.z = 1.0
        fov.color = ColorRGBA(r=0.0, g=0.9, b=0.46, a=0.12)
        edge = self.marker('radar_fov', 1, Marker.LINE_STRIP, RADAR_FRAME)
        edge.scale.x = 0.04
        edge.color = ColorRGBA(r=0.0, g=0.9, b=0.46, a=0.7)
        arc = [math.radians(-RADAR_HALF_AZ + 2 * RADAR_HALF_AZ * k / 30) for k in range(31)]
        z = -self.radar_height if self.room else -0.64
        pts = [Point(x=RADAR_RANGE * math.cos(a), y=RADAR_RANGE * math.sin(a), z=z) for a in arc]
        for p1, p2 in zip(pts, pts[1:]):
            fov.points += [Point(z=z), p1, p2]
        edge.points = [Point(z=z)] + pts + [Point(z=z)]
        arr.markers += [fov, edge]

        for i, (label, x, y) in enumerate([] if self.room else ROOMS):
            t = self.marker('rooms', i, Marker.TEXT_VIEW_FACING)
            t.text, t.scale.z = label, 0.6
            t.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0)
            t.pose.position.x, t.pose.position.y, t.pose.position.z = float(x), float(y), 0.1
            arr.markers.append(t)
        return arr


def main():
    rclpy.init()
    spin(SiteNode())


if __name__ == '__main__':
    main()
