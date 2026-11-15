"""End-to-end: the simulated demo (site, scripted B2, radar simulator) into the event log.

The real nodes run in one process: `site_node` publishes the building map and `base_link ->
radar_link`, `scripted_robot` patrols the hallway and publishes TF and `/joint_states`,
`radar_sim` casts the radar rays in its MuJoCo world, `detector` turns `/radar/points` into
`/alerts` and `/tracks`, `evaluator` scores them against `/sim/people`, and `event_logger` writes
the alerts to PostgreSQL. No launch file, no display, fixed seed.

Checked over 20 s of the scene: at least one alert, one alert per track, every alert inside the
building, the evaluator detects the walking person, and PostgreSQL holds exactly the alerts that
were published.
"""
import json
import math
import os
import time

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('mujoco')

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from rigsa_alpha.detector_node import DetectorNode  # noqa: E402
from rigsa_alpha.evaluator_node import Evaluator  # noqa: E402
from rigsa_alpha.event_logger_node import EventLogger  # noqa: E402
from rigsa_alpha.event_store import DEFAULT_DSN  # noqa: E402
from rigsa_alpha.radar_sim_node import RadarSim  # noqa: E402
from rigsa_alpha.scripted_robot_node import ScriptedRobot  # noqa: E402
from rigsa_alpha.site_node import SiteNode  # noqa: E402

try:
    import psycopg2
except ImportError:   # no driver: the event log side cannot be checked
    psycopg2 = None

PG_DSN = os.environ.get('RIGSA_TEST_DSN', DEFAULT_DSN)
SCHEMA = 'rigsa_demo'
SEED = 7             # fixed: the radar noise, ghost points and detections repeat
RUN_S = 20.0         # one person (starts at 4 s) walks into the radar's field of view
SETTLE_S = 2.0       # after the run: let the detector and the logger finish
STARTUP_S = 10.0     # map and TF before the first radar frame
# building bounds from floor.WALLS, for a sanity check on the alert geometry
X_LIMIT, Y_LIMIT = 14.0, 8.0


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


def ros_args(params):
    args = ['--ros-args']
    for k, v in params.items():
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        args += ['-p', '%s:=%s' % (k, v)]
    return args


class Harness(Node):
    """Subscriptions the test asserts on; the scene itself comes from the real nodes."""

    def __init__(self):
        super().__init__('demo_harness')
        self.alerts, self.tracks, self.evaluation = [], [], []
        self.point_clouds, self.points = 0, 0
        self.create_subscription(String, '/alerts', self.on_alert, 10)
        self.create_subscription(String, '/tracks', self.on_tracks, 10)
        self.create_subscription(String, '/evaluation', self.on_evaluation, 10)
        self.create_subscription(PointCloud2, '/radar/points', self.on_points, 10)

    def on_alert(self, msg):
        self.alerts.append(json.loads(msg.data))

    def on_tracks(self, msg):
        self.tracks.append(json.loads(msg.data))

    def on_evaluation(self, msg):
        self.evaluation.append(json.loads(msg.data))

    def on_points(self, msg):
        self.point_clouds += 1
        self.points += msg.width


def spin_for(executor, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


def wait_until(executor, ready, timeout):
    deadline = time.monotonic() + timeout
    while not ready():
        if time.monotonic() > deadline:
            return False
        executor.spin_once(timeout_sec=0.02)
    return True


@pytest.mark.skipif(not pg_available(), reason='PostgreSQL not reachable (set RIGSA_TEST_DSN)')
def test_simulated_demo_gives_alerts_and_rows_in_postgres(postgres_schema):
    # Arrange: the room test is off, so the detector uses its defaults; only the seed and the
    # event-log parameters are set
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('rigsa_alpha')   # site_node reads maps/ from the install space
    except Exception:
        pytest.skip('rigsa_alpha not installed (colcon build, then source install/setup.bash)')
    params = dict(gui=False, seed=SEED, robot_box=False, report_s=1000.0,
                  db_dsn=postgres_schema, sqlite_fallback=False, source='scripted')

    rclpy.init(args=ros_args(params))
    executor = SingleThreadedExecutor()
    nodes = []
    try:
        harness = Harness()
        site, robot, radar = SiteNode(), ScriptedRobot(), RadarSim()
        detector, evaluator, logger = DetectorNode(), Evaluator(), EventLogger()
        nodes += [harness, site, robot, radar, detector, evaluator, logger]
        for node in nodes:
            executor.add_node(node)

        # the map must reach the detector and the radar needs TF before its first scan
        assert wait_until(executor, lambda: detector.detector is not None
                          and harness.point_clouds > 0, STARTUP_S), (
            'startup failed: map %s, radar frames %d'
            % ('received' if detector.detector is not None else 'missing', harness.point_clouds))
        spin_for(executor, 0.5)      # let TF, the map and the first frames settle

        # Act: run the scene
        spin_for(executor, RUN_S)
        run_id = logger.run_id
        evaluator.report()          # score what happened, publish it on /evaluation
        spin_for(executor, SETTLE_S)
        logger.close()              # flush the buffer and close the run
        executor.spin_once(timeout_sec=0.1)

        # Assert: the radar simulator feeds the detector and the detector raises alerts
        assert harness.point_clouds >= 10.0 * RUN_S * 0.5, (
            '%d radar frames in %.0f s' % (harness.point_clouds, RUN_S))
        assert harness.points > 0, 'radar frames carried no points'
        assert harness.alerts, 'no alert in %.0f s of the scene' % RUN_S
        tracks = {tr['track'] for msg in harness.tracks for tr in msg['tracks']}
        alerted = [a['track'] for a in harness.alerts]
        assert set(alerted) <= tracks, 'alerted ids %r not in /tracks' % alerted
        assert len(set(alerted)) == len(alerted), 'one alert per person, got %r' % alerted
        for a in harness.alerts:
            assert abs(a['x']) < X_LIMIT and abs(a['y']) < Y_LIMIT, 'alert outside the building'
            assert 0.0 < a['distance'] < 15.0
            assert a['source'] == 'radar'

        # Assert: the evaluator scores the walking person against the simulated ground truth
        assert len(harness.evaluation) == 1, harness.evaluation
        score = harness.evaluation[0]
        assert score['alerts'] == len(harness.alerts)
        assert score['people_seen'] >= 1, 'nobody was visible to the radar in %.0f s' % RUN_S
        assert score['people_detected'] >= 1, score
        assert score['mean_delay_s'] is not None and score['mean_delay_s'] <= 5.0

        # Assert: PostgreSQL holds exactly the published alerts, with the same ids and geometry
        con = psycopg2.connect(postgres_schema, connect_timeout=2)
        try:
            with con.cursor() as c:
                c.execute('SELECT event_no, track_id, x_m, y_m, distance_m, robot_x, robot_y, '
                          'n_points, position_status, source FROM alerts WHERE run_id = %s '
                          'ORDER BY event_no', (run_id,))
                rows = c.fetchall()
                c.execute('SELECT source FROM patrol_runs WHERE id = %s', (run_id,))
                run = c.fetchone()
        finally:
            con.close()
        assert run == ('scripted',)
        assert [(r[0], r[1]) for r in rows] == [(a['event'], a['track']) for a in harness.alerts]
        for row, alert in zip(rows, harness.alerts):
            event_no, track_id, x, y, distance, robot_x, robot_y, n_points, status, source = row
            assert status == 'ok'
            assert source == 'radar'
            assert n_points > 0
            # the robot patrols, so the stored distance is from the robot pose, not from the map
            # origin (the bag test runs with the robot at the origin)
            assert distance == pytest.approx(math.hypot(x - robot_x, y - robot_y), abs=0.01)
            assert distance == pytest.approx(alert['distance'], abs=0.01)
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        rclpy.try_shutdown()
