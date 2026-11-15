"""End-to-end: a recorded radar bag through bag_radar, detector and event_logger into PostgreSQL.

The four real nodes run in one process with the room-mode TF and map from `site_node`: the radar
sits one metre above the map origin (the stand test), so the flat `z = 0` points of the AWR1443
land inside the detector's person height band.

Checked on `s2_one_walking`: exactly one alert, always the same track id, and one row in the
`alerts` table of the event log.
"""
import json
import math
import os
import time

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('rosbag2_py')

import rclpy  # noqa: E402
import yaml  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402

from b2_radar.bag_radar_node import BagRadar  # noqa: E402
from rigsa_alpha.common import RADAR_FRAME, RADAR_MOUNT, make_tf  # noqa: E402
from rigsa_alpha.detector_node import DetectorNode  # noqa: E402
from rigsa_alpha.event_logger_node import EventLogger  # noqa: E402
from rigsa_alpha.event_store import DEFAULT_DSN  # noqa: E402

try:
    import psycopg2
except ImportError:   # no driver: the event log side cannot be checked
    psycopg2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
BAG = os.path.join(HERE, '..', '..', '..', 'bags', 's2_one_walking')
PARAMS_YAML = os.path.join(HERE, '..', 'params', 'room_improved.yaml')
PG_DSN = os.environ.get('RIGSA_TEST_DSN', DEFAULT_DSN)
SCHEMA = 'rigsa_integration'
RADAR_HEIGHT = 1.0
# Playback speed. The bag is 47 s and must run in real time: with `rate` above 1 the Kalman gate
# (gate_chi2, kf_meas) is tighter for the same 0.1 m step between frames, so the tracker drops the
# person's blob and confirms a second track (checked at 2x, 4x and 8x: 2-3 alerts instead of 1).
RATE = 1.0
SETTLE_S = 2.0       # after the last frame: let the detector and the logger finish
REPLAY_TIMEOUT_S = 90
# Same tracking overrides as alpha.launch.py (robot:=none); the room tuning lives there because the
# real radar is sparser and jumpier than the simulated one.
ROOM_TRACKING = {'gate': 2.0, 'lost_after': 4.0, 'reacquire_s': 15.0, 'reacquire_m': 3.0,
                 'eps': 0.6}


def pg_available():
    if psycopg2 is None:
        return False
    try:
        psycopg2.connect(PG_DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


@pytest.fixture
def postgres_schema():
    """Empty schema for this test, dropped afterwards."""
    con = psycopg2.connect(PG_DSN, connect_timeout=2)
    con.autocommit = True
    with con.cursor() as c:
        c.execute('DROP SCHEMA IF EXISTS %s CASCADE' % SCHEMA)
        c.execute('CREATE SCHEMA %s' % SCHEMA)
    con.close()
    yield PG_DSN + ' options=-csearch_path=%s' % SCHEMA
    con = psycopg2.connect(PG_DSN, connect_timeout=2)
    con.autocommit = True
    with con.cursor() as c:
        c.execute('DROP SCHEMA IF EXISTS %s CASCADE' % SCHEMA)
    con.close()


def load_params(path):
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters']


def ros_args(params):
    args = ['--ros-args']
    for k, v in params.items():
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        args += ['-p', '%s:=%s' % (k, v)]
    return args


class Harness(Node):
    """Map, static TF and the subscriptions the test asserts on, standing in for site_node."""

    def __init__(self):
        super().__init__('integration_harness')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.alerts, self.tracks, self.status = [], [], []
        self.create_subscription(String, '/alerts', self.on_alert, 10)
        self.create_subscription(String, '/tracks', self.on_tracks, 10)
        self.create_subscription(String, '/event_log/status', self.on_status, 10)
        self.tf = StaticTransformBroadcaster(self)
        # site_node's room mode: the radar is at the map origin, RADAR_HEIGHT above the floor
        now = self.get_clock().now().to_msg()
        self.tf.sendTransform([
            make_tf(now, 'map', 'base_link',
                    (-RADAR_MOUNT[0], 0.0, RADAR_HEIGHT - RADAR_MOUNT[2])),
            make_tf(now, 'base_link', RADAR_FRAME, RADAR_MOUNT)])
        self.map_pub.publish(self.room_map())

    def room_map(self, res=0.05):
        """The all-free grid site_node publishes for the room test."""
        w, h = int(14.0 / res), int(16.0 / res)
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution, msg.info.width, msg.info.height = res, w, h
        msg.info.origin.position.x, msg.info.origin.position.y = -2.0, -8.0
        msg.data = [0] * (w * h)
        return msg

    def on_alert(self, msg):
        self.alerts.append(json.loads(msg.data))

    def on_tracks(self, msg):
        self.tracks.append(json.loads(msg.data))

    def on_status(self, msg):
        self.status.append(json.loads(msg.data))


@pytest.mark.skipif(not pg_available(), reason='PostgreSQL not reachable (set RIGSA_TEST_DSN)')
def test_one_walking_bag_gives_one_alert_and_one_track_in_postgres(postgres_schema):
    # Arrange: room-mode parameters, all passed as overrides so the real node classes are used
    assert os.path.isdir(BAG), 'bags/ not found; run the test from the repository'
    params = dict(ROOM_TRACKING)
    params.update(load_params(PARAMS_YAML))
    params.update(bag=os.path.abspath(BAG), loop=False, rate=RATE, room=True,
                  db_dsn=postgres_schema, sqlite_fallback=False, source='integration')

    rclpy.init(args=ros_args(params))
    executor = SingleThreadedExecutor()
    nodes = []
    try:
        harness, detector, logger = Harness(), DetectorNode(), EventLogger()
        nodes += [harness, detector, logger]
        for node in nodes:
            executor.add_node(node)

        # the static TF and the latched map must reach the detector before the bag starts
        deadline = time.monotonic() + 5.0
        while detector.detector is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert detector.detector is not None, 'detector never received /map'

        # Act: replay the whole bag once
        bag = BagRadar()
        nodes.append(bag)
        executor.add_node(bag)
        deadline = time.monotonic() + REPLAY_TIMEOUT_S
        while bag.i < len(bag.msgs) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert bag.i >= len(bag.msgs), 'bag replay did not finish within %d s' % REPLAY_TIMEOUT_S
        settle = time.monotonic() + SETTLE_S
        while time.monotonic() < settle:
            executor.spin_once(timeout_sec=0.02)
        run_id = logger.run_id
        logger.close()       # flush the buffer and close the run
        executor.spin_once(timeout_sec=0.1)

        # Assert: exactly one alert, published by the detector
        assert len(harness.alerts) == 1, 'expected one alert, got %r' % (harness.alerts,)
        alert = harness.alerts[0]
        assert alert['event'] == 1
        assert alert['source'] == 'radar'
        assert 0.0 < alert['distance'] < 10.0

        # Assert: one track id for the whole run, the one in the alert
        seen = {tr['track'] for msg in harness.tracks for tr in msg['tracks']}
        assert seen == {alert['track']}
        tracked = sum(any(tr['track'] == alert['track'] for tr in msg['tracks'])
                      for msg in harness.tracks)
        assert tracked >= 0.8 * len(harness.tracks), (
            'track id %s only in %d / %d frames' % (alert['track'], tracked,
                                                   len(harness.tracks)))

        # Assert: the alert reached PostgreSQL, once, with the detector's geometry
        con = psycopg2.connect(postgres_schema, connect_timeout=2)
        try:
            with con.cursor() as c:
                c.execute('SELECT event_no, track_id, x_m, y_m, distance_m, n_points, '
                          'position_status, source FROM alerts WHERE run_id = %s', (run_id,))
                rows = c.fetchall()
                c.execute('SELECT source FROM patrol_runs WHERE id = %s', (run_id,))
                run = c.fetchone()
        finally:
            con.close()
        assert run == ('integration',)
        assert len(rows) == 1
        event_no, track_id, x, y, distance, n_points, status, source = rows[0]
        assert (event_no, track_id, source) == (1, alert['track'], 'radar')
        assert status == 'ok'
        assert n_points > 0
        assert distance == pytest.approx(math.hypot(x, y), abs=0.01)   # radar at the map origin
        assert distance == pytest.approx(alert['distance'], abs=0.01)
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        rclpy.try_shutdown()
