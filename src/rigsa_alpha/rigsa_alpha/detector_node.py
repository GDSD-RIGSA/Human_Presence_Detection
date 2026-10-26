"""Detection & Alerts container: finds people in the radar point cloud and raises alerts.

Uses only radar data, the map and the robot pose (TF). It works the same on simulated points,
a rosbag recording or the real TI driver (remap its topic to /radar/points).

Subscribes: /radar/points (PointCloud2, any frame with TF to map), /map (OccupancyGrid), /alerts/ack (JSON)
Publishes:  /alerts (JSON, one per newly confirmed person), /tracks (JSON, confirmed tracks at radar rate),
            /tracks/markers (MarkerArray for RViz), /radar/foreground (PointCloud2, points left after filtering)
"""
import json
import math
from collections import deque
from datetime import datetime, timezone

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rigsa_alpha.common import make_cloud, quat_to_yaw, read_cloud_full, spin, transform_matrix
from rigsa_alpha.detection import EPS, GATE, LOST_AFTER, REACQUIRE_M, REACQUIRE_S, BackgroundMask, Detector

RED = ColorRGBA(r=0.90, g=0.22, b=0.21, a=0.9)
GREY = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.9)
# one colour per person (by alert number), so several people are easy to tell apart in RViz
PERSON_COLOURS = [ColorRGBA(r=r, g=g, b=b, a=0.9) for r, g, b in (
    (0.12, 0.47, 0.90), (0.18, 0.70, 0.30), (0.98, 0.60, 0.00), (0.58, 0.25, 0.80),
    (0.00, 0.70, 0.75), (0.95, 0.80, 0.10), (0.95, 0.35, 0.65), (0.55, 0.40, 0.25))]   # no red: red = new alert ring
STATE_TEXT = {'new': '', 'ack': ' (ack)', 'false_alarm': ' (false alarm)'}
MAX_TILT_DEG = 30   # frames from a tilted radar (fallen robot) see floor and ceiling, not people
# Detector defaults, for logging which parameters differ
DEFAULT_TUNING = {'gate': GATE, 'lost_after': LOST_AFTER, 'reacquire_s': REACQUIRE_S, 'reacquire_m': REACQUIRE_M,
                  'eps': EPS, 'max_range': 0.0, 'snr_db': 0.0, 'snr_db_per_m': 0.0, 'min_points_per_m': 0.0,
                  'min_points_r0': 2.0, 'doppler_k': 0.0, 'tracker': 'greedy', 'kf_accel': 1.5, 'kf_meas': 0.3,
                  'gate_chi2': 9.21, 'absorb_m': 0.0, 'static_hold_s': 0.0, 'static_speed': 0.6,
                  'static_radius': 1.5, 'new_track_min_points': 0}
MAX_WAIT = 3.0   # s, longest a cloud waits for its robot pose before the latest pose is used


class DetectorNode(Node):
    def __init__(self):
        super().__init__('rigsa_detector')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        # tracking overrides (defaults = the tuning in detection.py); the room test with the real radar loosens them
        self.tuning = {k: self.declare_parameter(k, v).value for k, v in (
            ('gate', GATE), ('lost_after', LOST_AFTER), ('reacquire_s', REACQUIRE_S), ('reacquire_m', REACQUIRE_M),
            ('eps', EPS))}
        # room-mode options of the improved detector (params/room_improved.yaml); the defaults switch them off
        self.tuning.update({k: self.declare_parameter(k, v).value for k, v in (
            ('max_range', 0.0), ('snr_db', 0.0), ('snr_db_per_m', 0.0), ('min_points_per_m', 0.0),
            ('min_points_r0', 2.0), ('doppler_k', 0.0), ('tracker', 'greedy'), ('kf_accel', 1.5), ('kf_meas', 0.3),
            ('gate_chi2', 9.21), ('absorb_m', 0.0), ('static_hold_s', 0.0), ('static_speed', 0.6),
            ('static_radius', 1.5), ('new_track_min_points', 0))})
        changed = {k: v for k, v in self.tuning.items() if v != DEFAULT_TUNING.get(k, v)}
        self.get_logger().info('Detector parameters: %s' % (', '.join('%s=%s' % kv for kv in sorted(changed.items()))
                                                            or 'defaults'))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detector = None
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(PointCloud2, '/radar/points', self.on_points, qos_profile_sensor_data)
        self.create_subscription(String, '/alerts/ack', self.on_ack, 10)
        self.alert_pub = self.create_publisher(String, '/alerts', 10)
        self.tracks_pub = self.create_publisher(String, '/tracks', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tracks/markers', 10)
        self.fg_pub = self.create_publisher(PointCloud2, '/radar/foreground', 10)
        self.pending = deque()   # clouds waiting for the robot pose at their timestamp
        self.warned = False
        self.get_logger().info('Detector waiting for /map and /radar/points')

    def on_map(self, msg):
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.detector = Detector(BackgroundMask(grid, msg.info.resolution, origin), **self.tuning)
        self.get_logger().info('Map received (%dx%d); detector ready' % (msg.info.width, msg.info.height))

    def lookup(self, frame, stamp, exact_only=False):
        try:
            return self.tf_buffer.lookup_transform('map', frame, Time.from_msg(stamp))
        except TransformException:
            if exact_only:
                return None
            try:   # fall back to the latest transform (e.g. clock differences when replaying a bag)
                return self.tf_buffer.lookup_transform('map', frame, Time())
            except TransformException as e:
                if not self.warned:
                    self.get_logger().warn('No TF map -> %s yet: %s' % (frame, e))
                    self.warned = True
                return None

    def on_points(self, msg):
        """Clouds are processed in order, each with the robot pose at its own timestamp. A cloud whose pose has not
        arrived yet waits (up to MAX_WAIT); this matters when replaying a bag, where TF can trail the radar."""
        if self.detector is None:
            return
        self.pending.append(msg)
        while self.pending:
            head = self.pending[0]
            if not self.pose_ready(head):
                break
            self.process(self.pending.popleft())

    def pose_ready(self, msg):
        """True when msg can be processed: its pose is known, will never arrive (TF is already newer), or the
        cloud has waited MAX_WAIT."""
        if self.lookup(msg.header.frame_id, msg.header.stamp, exact_only=True) is not None:
            return True
        stamp = Time.from_msg(msg.header.stamp)
        try:
            if Time.from_msg(self.tf_buffer.lookup_transform('map', 'base_link', Time()).header.stamp) > stamp:
                return True
        except TransformException:
            pass
        newest = Time.from_msg(self.pending[-1].header.stamp)
        return (newest - stamp).nanoseconds * 1e-9 >= MAX_WAIT

    def process(self, msg):
        tf_radar = self.lookup(msg.header.frame_id, msg.header.stamp)
        tf_robot = self.lookup('base_link', msg.header.stamp)
        if tf_radar is None or tf_robot is None:
            return
        T = transform_matrix(tf_radar.transform)
        if T[2, 2] < math.cos(math.radians(MAX_TILT_DEG)):   # radar's up axis far from vertical: the B2 has fallen
            self.get_logger().warn('Radar tilted more than %d deg (has the B2 fallen?); ignoring its points'
                                   % MAX_TILT_DEG, throttle_duration_sec=5.0)
            return
        pts = read_cloud_full(msg)   # x, y, z, intensity, velocity (radial velocity does not change with the frame)
        if len(pts):
            pts[:, :3] = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        new = self.detector.update(pts, t, sensor_xy=T[:2, 3])

        r = tf_robot.transform
        robot = (r.translation.x, r.translation.y, quat_to_yaw(r.rotation))
        for tr in new:
            self.raise_alert(tr, robot, msg.header.stamp)
        self.publish_tracks(robot, msg.header)
        fg = self.detector.frames[-1] if self.detector.frames else np.zeros((0, 5))
        cloud = np.zeros((len(fg), 5), dtype=np.float32)
        cloud[:, :fg.shape[1]] = fg
        self.fg_pub.publish(make_cloud(cloud, msg.header.stamp, frame='map'))

    def relative(self, tr, robot):
        dx, dy = tr.pos[0] - robot[0], tr.pos[1] - robot[1]
        bearing = math.degrees(math.atan2(dy, dx) - robot[2])
        bearing = (bearing + 180.0) % 360.0 - 180.0
        return math.hypot(dx, dy), bearing

    def raise_alert(self, tr, robot, stamp):
        dist, bearing = self.relative(tr, robot)
        side = 'left' if bearing >= 0 else 'right'
        when = datetime.fromtimestamp(Time.from_msg(stamp).nanoseconds * 1e-9, tz=timezone.utc).astimezone()
        msg = dict(event=tr.event, track=tr.id, distance=round(dist, 2), bearing_deg=round(bearing, 1),
                   direction='%d° %s' % (abs(round(bearing)), side), x=round(float(tr.pos[0]), 2),
                   y=round(float(tr.pos[1]), 2), n_points=int(tr.n_points),
                   robot_x=round(robot[0], 2), robot_y=round(robot[1], 2), robot_yaw=round(robot[2], 3),
                   time=when.strftime('%H:%M:%S'), stamp=when.isoformat(), source='radar')
        self.alert_pub.publish(String(data=json.dumps(msg)))
        self.get_logger().info('ALERT %d: person %s at %.1f m, %s (%d radar points)'
                               % (tr.event, tr.id, dist, msg['direction'], tr.n_points))

    def on_ack(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if self.detector is not None and d.get('action') in ('ack', 'false_alarm'):
            self.detector.set_state(d.get('event'), d['action'])

    def publish_tracks(self, robot, header):
        tracks = []
        for tr in self.detector.confirmed_tracks():
            dist, bearing = self.relative(tr, robot)
            tracks.append(dict(track=tr.id, event=tr.event, x=round(float(tr.pos[0]), 2), y=round(float(tr.pos[1]), 2),
                               distance=round(dist, 2), state=tr.state))
        stamp = Time.from_msg(header.stamp).nanoseconds * 1e-9
        self.tracks_pub.publish(String(data=json.dumps(dict(stamp=stamp, tracks=tracks))))
        self.marker_pub.publish(self.markers(robot))

    def marker(self, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def markers(self, robot):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        k = 0
        for tr in self.detector.tracks:
            x, y = float(tr.pos[0]), float(tr.pos[1])
            if not tr.confirmed:   # candidate: small grey dot
                dot = self.marker('candidates', k, Marker.SPHERE)
                dot.scale.x = dot.scale.y = dot.scale.z = 0.3
                dot.pose.position.x, dot.pose.position.y, dot.pose.position.z = x, y, 0.15
                dot.color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.5)
                arr.markers.append(dot)
                k += 1
                continue
            col = GREY if tr.state == 'false_alarm' else PERSON_COLOURS[(tr.event - 1) % len(PERSON_COLOURS)]
            body = self.marker('people', k, Marker.CYLINDER)
            body.scale.x = body.scale.y = 0.6
            body.scale.z = 1.7
            body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, 0.85
            body.color = col
            label = self.marker('people', k + 1, Marker.TEXT_VIEW_FACING)
            label.text = '%s  %.1f m%s%s' % (tr.id, math.hypot(x - robot[0], y - robot[1]),
                                             ' (still)' if tr.life == 'static' else '', STATE_TEXT.get(tr.state, ''))
            label.scale.z, label.color = 0.55, ColorRGBA(r=col.r * 0.7, g=col.g * 0.7, b=col.b * 0.7, a=1.0)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x + 1.2, y + 0.6, 2.0
            trail = self.marker('people', k + 2, Marker.LINE_STRIP)
            trail.scale.x, trail.color = 0.08, col
            trail.points = [Point(x=float(px), y=float(py), z=0.05) for px, py in tr.trail + [(x, y)]]
            arr.markers += [body, label, trail]
            if tr.state == 'new':
                ring = self.marker('people', k + 3, Marker.LINE_STRIP)
                ring.scale.x, ring.color = 0.1, RED
                ring.points = [Point(x=x + 0.9 * math.cos(a / 20 * 2 * math.pi), y=y + 0.9 * math.sin(a / 20 * 2 * math.pi),
                                     z=0.05) for a in range(21)]
                alert = self.marker('people', k + 4, Marker.TEXT_VIEW_FACING)
                alert.text, alert.scale.z, alert.color = 'PERSON DETECTED', 0.7, ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0)
                alert.pose.position.x, alert.pose.position.y, alert.pose.position.z = x, y - 1.3, 2.0
                arr.markers += [ring, alert]
            k += 10
        return arr


def main():
    rclpy.init()
    node = DetectorNode()
    spin(node)


if __name__ == '__main__':
    main()
