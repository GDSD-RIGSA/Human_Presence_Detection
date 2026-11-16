"""Logic inside the ROS nodes (alert geometry, message handling, scoring) without a ROS graph.

Each node method is borrowed onto a small fake, as in test_radar_driver.py, so no rclpy.init(),
topics or timers are needed.
"""
import json
import math
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('tf2_ros')
from builtin_interfaces.msg import Time  # noqa: E402
from rigsa_alpha.detector_node import DetectorNode  # noqa: E402
from rigsa_alpha.evaluator_node import FALSE_M, Evaluator  # noqa: E402
from rigsa_alpha.event_logger_node import EventLogger  # noqa: E402


class FakeLogger:
    """Records (level, text); accepts the throttle_duration_sec keyword like rclpy."""

    def __init__(self):
        self.lines = []

    def _add(self, level):
        return lambda text, **kw: self.lines.append((level, text))

    def __getattr__(self, level):
        return self._add(level)


def msg(payload):
    return SimpleNamespace(data=payload if isinstance(payload, str) else json.dumps(payload))


# ---------------------------------------------------------------- detector_node

class FakeDetectorNode:
    relative = DetectorNode.relative
    raise_alert = DetectorNode.raise_alert
    on_ack = DetectorNode.on_ack

    def __init__(self, detector='present'):
        self.sent = []
        self.states = []
        self.alert_pub = SimpleNamespace(publish=self.sent.append)
        self.log = FakeLogger()
        self.detector = (SimpleNamespace(set_state=lambda e, s: self.states.append((e, s)))
                         if detector else None)

    def get_logger(self):
        return self.log


def track(x, y, event=1, tid=3, n_points=12):
    return SimpleNamespace(pos=(x, y), event=event, id=tid, n_points=n_points)


def test_relative_person_straight_ahead():
    # Arrange
    node = FakeDetectorNode()
    robot = (0.0, 0.0, 0.0)
    # Act
    dist, bearing = node.relative(track(3.0, 0.0), robot)
    # Assert
    assert dist == pytest.approx(3.0)
    assert bearing == pytest.approx(0.0)


def test_relative_bearing_follows_the_robot_heading():
    # Arrange: robot faces +y, person is 2 m along +y
    node = FakeDetectorNode()
    robot = (0.0, 0.0, math.pi / 2)
    # Act
    dist, bearing = node.relative(track(0.0, 2.0), robot)
    # Assert
    assert dist == pytest.approx(2.0)
    assert bearing == pytest.approx(0.0)


def test_relative_bearing_directly_behind_wraps_to_minus_180():
    # Arrange: boundary of the wrap, atan2 gives +180 and the range is [-180, 180)
    node = FakeDetectorNode()
    # Act
    _, bearing = node.relative(track(-2.0, 0.0), (0.0, 0.0, 0.0))
    # Assert
    assert bearing == pytest.approx(-180.0)


def test_relative_bearing_wraps_across_the_180_line():
    # Arrange: robot yaw 170 deg, person at absolute -170 deg, i.e. 20 deg to the robot's left
    node = FakeDetectorNode()
    ang = math.radians(-170)
    # Act
    _, bearing = node.relative(track(2 * math.cos(ang), 2 * math.sin(ang)),
                               (0.0, 0.0, math.radians(170)))
    # Assert
    assert bearing == pytest.approx(20.0)


def test_relative_person_on_the_robot_has_zero_distance():
    # Arrange / Act
    dist, bearing = FakeDetectorNode().relative(track(1.0, 1.0), (1.0, 1.0, 0.0))
    # Assert: no division by zero, atan2(0, 0) is 0
    assert dist == 0.0
    assert bearing == 0.0


def test_raise_alert_publishes_one_json_alert_with_geometry():
    # Arrange
    node = FakeDetectorNode()
    robot = (1.0, 1.0, 0.0)
    # Act
    node.raise_alert(track(1.0, 3.0, event=7, tid=4, n_points=15), robot, Time(sec=100))
    # Assert
    assert len(node.sent) == 1
    a = json.loads(node.sent[0].data)
    assert (a['event'], a['track'], a['n_points'], a['source']) == (7, 4, 15, 'radar')
    assert a['distance'] == 2.0
    assert a['bearing_deg'] == 90.0
    assert a['direction'] == '90° left'
    assert (a['x'], a['y'], a['robot_x'], a['robot_y']) == (1.0, 3.0, 1.0, 1.0)
    assert node.log.lines[0][0] == 'info'


@pytest.mark.parametrize('y, direction', [(0.0, '0° left'), (-2.0, '90° right')])
def test_raise_alert_side_boundary(y, direction):
    # Arrange: bearing exactly 0 counts as left, any negative bearing is right
    node = FakeDetectorNode()
    # Act
    node.raise_alert(track(2.0 if y == 0.0 else 0.0, y), (0.0, 0.0, 0.0), Time(sec=1))
    # Assert
    assert json.loads(node.sent[0].data)['direction'] == direction


@pytest.mark.parametrize('action', ['ack', 'false_alarm'])
def test_on_ack_sets_track_state(action):
    # Arrange
    node = FakeDetectorNode()
    # Act
    node.on_ack(msg(dict(event=5, action=action)))
    # Assert
    assert node.states == [(5, action)]


@pytest.mark.parametrize('payload', ['not json', json.dumps(dict(event=5, action='delete')),
                                     json.dumps(dict(event=5))])
def test_on_ack_ignores_bad_or_unknown_actions(payload):
    # Arrange
    node = FakeDetectorNode()
    # Act
    node.on_ack(msg(payload))
    # Assert
    assert node.states == []


def test_on_ack_before_the_map_arrives_is_ignored():
    # Arrange: detector is created when /map arrives, so it is None until then
    node = FakeDetectorNode(detector=None)
    # Act
    node.on_ack(msg(dict(event=5, action='ack')))
    # Assert
    assert node.states == []


# ----------------------------------------------------------- event_logger_node

class FakeWriter:
    def __init__(self, reply=()):
        self.reply = list(reply)
        self.alerts = []
        self.actions = []

    def submit_alert(self, a):
        self.alerts.append(a)
        return self.reply

    def submit_action(self, d):
        self.actions.append(d)
        return self.reply


class FakeLoggerNode:
    log = EventLogger.log
    on_alert = EventLogger.on_alert
    on_ack = EventLogger.on_ack

    def __init__(self, reply=()):
        self.writer = FakeWriter(reply)
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger


def test_logger_forwards_a_valid_alert_to_the_writer():
    # Arrange
    node = FakeLoggerNode()
    alert = dict(event=1, x=1.0, y=2.0)
    # Act
    node.on_alert(msg(alert))
    # Assert
    assert node.writer.alerts == [alert]
    assert node.logger.lines == []


def test_logger_rejects_an_alert_that_is_not_json():
    # Arrange
    node = FakeLoggerNode()
    # Act
    node.on_alert(msg('{broken'))
    # Assert
    assert node.writer.alerts == []
    assert node.logger.lines[0][0] == 'error'
    assert 'not JSON' in node.logger.lines[0][1]


def test_logger_forwards_an_operator_action_and_rejects_bad_json():
    # Arrange
    node = FakeLoggerNode()
    # Act
    node.on_ack(msg(dict(event=2, action='ack')))
    node.on_ack(msg(''))
    # Assert
    assert node.writer.actions == [dict(event=2, action='ack')]
    assert [level for level, _ in node.logger.lines] == ['error']


def test_logger_logs_the_writers_messages_at_their_level():
    # Arrange
    node = FakeLoggerNode(reply=[('warn', 'a'), ('info', 'b')])
    # Act
    node.on_alert(msg(dict(event=1)))
    # Assert
    assert node.logger.lines == [('warn', 'a'), ('info', 'b')]


# --------------------------------------------------------------- evaluator_node

class FakeEvaluator:
    on_truth = Evaluator.on_truth
    on_alert = Evaluator.on_alert
    report = Evaluator.report

    def __init__(self):
        self.truth = []
        self.current = {}
        self.appearances = []
        self.alerts = 0
        self.false_alerts = 0
        self.sent = []
        self.pub = SimpleNamespace(publish=self.sent.append)
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger


def appearance(start, last, detected_at=None):
    return dict(person=1, start=start, last=last, detected_at=detected_at)


def truth(stamp, *people):
    return msg(dict(stamp=stamp, people=[dict(person=i, x=x, y=y, visible=v)
                                         for i, x, y, v in people]))


def test_report_with_no_appearances_has_no_mean_delay():
    # Arrange
    ev = FakeEvaluator()
    # Act
    ev.report()
    # Assert: empty input must not divide by zero
    res = json.loads(ev.sent[0].data)
    assert res == dict(people_seen=0, people_detected=0, alerts=0, false_alerts=0,
                       mean_delay_s=None)


@pytest.mark.parametrize('last, seen', [(10.9, 0), (11.0, 1)])
def test_report_counts_a_person_after_one_second_visible(last, seen):
    # Arrange: visible 10.0 -> last, the 1.0 s limit is inclusive
    ev = FakeEvaluator()
    ev.appearances = [appearance(10.0, last)]
    # Act
    ev.report(publish=False)
    # Assert
    assert 'Evaluation: 0/%d people detected' % seen in ev.logger.lines[0][1]


def test_report_mean_delay_and_detection_rate():
    # Arrange: two seen people, one detected after 2 s, one never; a short glimpse is ignored
    ev = FakeEvaluator()
    ev.appearances = [appearance(0.0, 10.0, detected_at=2.0), appearance(0.0, 10.0),
                      appearance(5.0, 5.5, detected_at=5.2)]
    ev.alerts = 3
    # Act
    ev.report()
    # Assert
    res = json.loads(ev.sent[0].data)
    assert (res['people_seen'], res['people_detected'], res['alerts']) == (2, 1, 3)
    assert res['mean_delay_s'] == 2.0


def test_on_truth_starts_one_appearance_per_visible_person():
    # Arrange
    ev = FakeEvaluator()
    # Act
    ev.on_truth(truth(1.0, (1, 0.0, 0.0, True), (2, 5.0, 5.0, False)))
    ev.on_truth(truth(2.0, (1, 0.0, 0.0, True)))
    # Assert: person 2 was never visible, person 1 is one appearance from 1.0 to 2.0
    assert [(a['person'], a['start'], a['last']) for a in ev.appearances] == [(1, 1.0, 2.0)]


@pytest.mark.parametrize('gap, forgotten', [(2.0, False), (2.1, True)])
def test_on_truth_forgets_a_person_out_of_view_for_more_than_two_seconds(gap, forgotten):
    # Arrange
    ev = FakeEvaluator()
    ev.on_truth(truth(10.0, (1, 0.0, 0.0, True)))
    # Act
    ev.on_truth(truth(10.0 + gap, (1, 0.0, 0.0, False)))
    # Assert
    assert (1 not in ev.current) is forgotten


def test_on_truth_new_sighting_after_a_gap_is_a_new_appearance():
    # Arrange
    ev = FakeEvaluator()
    ev.on_truth(truth(10.0, (1, 0.0, 0.0, True)))
    ev.on_truth(truth(13.0, (1, 0.0, 0.0, False)))
    # Act
    ev.on_truth(truth(14.0, (1, 0.0, 0.0, True)))
    # Assert
    assert len(ev.appearances) == 2


@pytest.mark.parametrize('dist, false_alerts', [(FALSE_M, 0), (FALSE_M + 0.01, 1)])
def test_on_alert_false_alert_boundary(dist, false_alerts):
    # Arrange: a person at the origin; exactly FALSE_M away still counts as near
    ev = FakeEvaluator()
    ev.truth = [dict(person=1, x=0.0, y=0.0, visible=True)]
    # Act
    ev.on_alert(msg(dict(event=1, x=dist, y=0.0)))
    # Assert
    assert (ev.alerts, ev.false_alerts) == (1, false_alerts)


def test_on_alert_with_nobody_in_the_simulation_is_false():
    # Arrange
    ev = FakeEvaluator()
    # Act
    ev.on_alert(msg(dict(event=4, x=1.0, y=1.0)))
    # Assert
    assert ev.false_alerts == 1
    assert ev.logger.lines[0][0] == 'warn'
