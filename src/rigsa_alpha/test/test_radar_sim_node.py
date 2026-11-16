"""Radar simulator node: TF lookup, people velocities, robot placement, ground truth, step.

Node methods are borrowed onto a fake, so no ROS graph or MuJoCo viewer is needed. Skipped where
`mujoco` (imported by the node's module) or `tf2_ros` is missing.
"""
import contextlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('tf2_ros')
pytest.importorskip('mujoco')
from builtin_interfaces.msg import Time  # noqa: E402
from rigsa_alpha import scenario  # noqa: E402
from rigsa_alpha.common import make_tf, read_cloud_full, transform_matrix, yaw_quat  # noqa: E402
from rigsa_alpha.radar_sim_node import RADAR_HZ, RADAR_RANGE, RadarSim  # noqa: E402
from tf2_ros import TransformException  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

EYE = np.eye(3)
ORIGIN = np.array([0.0, 0.0, 1.0])


class FakeLogger:
    def __init__(self):
        self.lines = []

    def warn(self, text, **kw):
        self.lines.append(('warn', text))

    def info(self, text, **kw):
        self.lines.append(('info', text))


class FakeWorld:
    def __init__(self, sight=True):
        self.sight = sight
        self.calls = []
        self.sight_calls = []

    def set_people(self, people, vel):
        self.calls.append(('people', people, vel))

    def set_robot(self, pos, quat, joints):
        self.calls.append(('robot', list(pos), list(quat), dict(joints)))

    def forward(self):
        self.calls.append(('forward',))

    def radar_scan(self, origin, rot, sensor_vel):
        self.calls.append(('scan', np.array(origin), np.array(rot), np.array(sensor_vel)))
        return np.array([[1.0, 0.0, 1.0, 20.0, 0.5], [2.0, 1.0, 1.0, 18.0, -0.5]],
                        dtype=np.float32)

    def line_of_sight(self, origin, target):
        self.sight_calls.append(target)
        return self.sight


class FakeSim:
    on_joints = RadarSim.on_joints
    lookup = RadarSim.lookup
    people = RadarSim.people
    set_robot = RadarSim.set_robot
    world_lock = RadarSim.world_lock
    draw = RadarSim.draw
    publish_truth = RadarSim.publish_truth
    step = RadarSim.step

    def __init__(self, world=None):
        self.world = world or FakeWorld()
        self.joints = {}
        self.prev = None
        self.warned = False
        self.viewer = None
        self.last_origin = None
        self.t0 = 0.0
        self.log = FakeLogger()
        self.tf_buffer = SimpleNamespace(lookup_transform=self._lookup)
        self.tf_msgs = {}
        self.truth, self.markers, self.points = [], [], []
        self.truth_pub = SimpleNamespace(publish=self.truth.append)
        self.truth_markers = SimpleNamespace(publish=self.markers.append)
        self.points_pub = SimpleNamespace(publish=self.points.append)

    def get_logger(self):
        return self.log

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=7_500_000_000))

    def _lookup(self, target, frame, when, timeout=None):
        if frame not in self.tf_msgs:
            raise TransformException('no transform to %s' % frame)
        return self.tf_msgs[frame]


def with_tf(sim, **frames):
    """frames: name -> (x, y, z, yaw)."""
    for name, (x, y, z, yaw) in frames.items():
        sim.tf_msgs[name] = make_tf(Time(sec=5), 'map', name, (x, y, z), yaw_quat(yaw))
    return sim


# ---------------------------------------------------------------------------- joints

def test_on_joints_keeps_name_to_angle_pairs():
    # Arrange
    sim = FakeSim()
    # Act
    sim.on_joints(SimpleNamespace(name=['a', 'b'], position=[0.1, 0.2]))
    # Assert
    assert sim.joints == {'a': 0.1, 'b': 0.2}


def test_on_joints_replaces_the_previous_message():
    # Arrange
    sim = FakeSim()
    sim.on_joints(SimpleNamespace(name=['a'], position=[1.0]))
    # Act
    sim.on_joints(SimpleNamespace(name=['b'], position=[2.0]))
    # Assert
    assert sim.joints == {'b': 2.0}


# ----------------------------------------------------------------------------- lookup

def test_lookup_returns_the_transform_matrix():
    # Arrange
    sim = with_tf(FakeSim(), base_link=(1.0, 2.0, 0.5, math.pi / 2))
    # Act
    T = sim.lookup('base_link')
    # Assert
    assert T[:3, 3] == pytest.approx([1.0, 2.0, 0.5])
    assert T[:3, :3] @ [1.0, 0, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_lookup_with_stamp_also_returns_the_pose_time():
    # Arrange
    sim = with_tf(FakeSim(), radar_link=(0.0, 0.0, 1.0, 0.0))
    # Act
    T, stamp = sim.lookup('radar_link', stamp=True)
    # Assert
    assert T is not None
    assert stamp.sec == 5


@pytest.mark.parametrize('stamp, expected', [(False, None), (True, (None, None))])
def test_lookup_without_a_transform_returns_none(stamp, expected):
    # Arrange
    sim = FakeSim()
    # Act
    result = sim.lookup('base_link', stamp=stamp)
    # Assert
    assert result == expected


def test_lookup_warns_only_once_while_the_robot_pose_is_missing():
    # Arrange
    sim = FakeSim()
    # Act
    for _ in range(3):
        sim.lookup('base_link')
    # Assert
    assert [lvl for lvl, _ in sim.log.lines] == ['warn']
    assert 'base_link' in sim.log.lines[0][1]


# ----------------------------------------------------------------------------- people

def test_people_first_call_has_no_velocity():
    # Arrange: person 0 is already walking at t = 10
    sim = FakeSim()
    # Act
    people, vel = sim.people(10.0, 0.1)
    # Assert: no previous frame to difference against
    assert people[0] is not None
    assert vel.shape == (len(scenario.PEOPLE), 3)
    assert not vel.any()


def test_people_velocity_is_the_displacement_over_dt():
    # Arrange: person 0 walks a straight stretch at the scenario's walking speed
    sim = FakeSim()
    sim.people(scenario.PEOPLE[0]['start'] + 1.0, 0.1)
    # Act
    _, vel = sim.people(scenario.PEOPLE[0]['start'] + 1.1, 0.1)
    # Assert
    assert np.hypot(*vel[0, :2]) == pytest.approx(scenario.WALK, abs=1e-6)
    assert vel[0, 2] == 0.0


def test_people_who_are_not_in_the_scene_have_zero_velocity():
    # Arrange: before anyone has started, everyone is None
    sim = FakeSim()
    sim.people(0.0, 0.1)
    # Act
    people, vel = sim.people(0.1, 0.1)
    # Assert
    assert people == [None] * len(scenario.PEOPLE)
    assert not vel.any()


def test_people_person_who_just_appeared_has_zero_velocity():
    # Arrange: person 0 not yet started, then appears
    sim = FakeSim()
    sim.people(scenario.PEOPLE[0]['start'] - 0.05, 0.1)
    # Act
    people, vel = sim.people(scenario.PEOPLE[0]['start'], 0.1)
    # Assert
    assert people[0] is not None
    assert not vel[0].any()


# --------------------------------------------------------------------------- set_robot

def test_set_robot_passes_position_identity_quaternion_and_joints():
    # Arrange
    sim = with_tf(FakeSim(), base_link=(1.0, 2.0, 0.5, 0.0))
    sim.joints = {'FR_hip_joint': 0.2}
    # Act
    sim.set_robot(sim.lookup('base_link'))
    # Assert
    _, pos, quat, joints = sim.world.calls[0]
    assert pos == pytest.approx([1.0, 2.0, 0.5])
    assert quat == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert joints == {'FR_hip_joint': 0.2}


@pytest.mark.parametrize('yaw', [0.4, math.pi / 2, -2.0])
def test_set_robot_converts_the_rotation_matrix_to_a_wxyz_quaternion(yaw):
    # Arrange
    sim = FakeSim()
    T = transform_matrix(make_tf(Time(), 'map', 'b', (0, 0, 0), yaw_quat(yaw)).transform)
    # Act
    sim.set_robot(T)
    # Assert
    quat = sim.world.calls[0][2]
    assert quat == pytest.approx([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], abs=1e-9)


def test_set_robot_half_turn_does_not_divide_by_zero():
    # Arrange: trace of the rotation is -1, so w is about 0 (the code floors it)
    sim = FakeSim()
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    # Act
    sim.set_robot(T)
    # Assert
    quat = sim.world.calls[0][2]
    assert all(math.isfinite(v) for v in quat)


# ------------------------------------------------------------ world_lock and draw

def test_world_lock_without_a_viewer_is_a_no_op_context():
    # Arrange
    sim = FakeSim()
    # Act / Assert
    with sim.world_lock():
        pass
    assert isinstance(sim.world_lock(), contextlib.nullcontext)


def test_world_lock_with_a_viewer_uses_the_viewer_lock():
    # Arrange
    sentinel = contextlib.nullcontext('viewer lock')
    sim = FakeSim()
    sim.viewer = SimpleNamespace(lock=lambda: sentinel)
    # Act / Assert
    assert sim.world_lock() is sentinel


def test_draw_without_a_viewer_does_nothing():
    # Arrange / Act / Assert: must not raise
    FakeSim().draw()


def test_draw_syncs_a_running_viewer():
    # Arrange
    calls = []
    sim = FakeSim()
    sim.viewer = SimpleNamespace(is_running=lambda: True, sync=lambda: calls.append('sync'))
    # Act
    sim.draw()
    # Assert
    assert calls == ['sync']
    assert sim.viewer is not None


def test_draw_drops_a_viewer_that_was_closed():
    # Arrange
    calls = []
    sim = FakeSim()
    sim.viewer = SimpleNamespace(is_running=lambda: False, sync=lambda: calls.append('sync'))
    # Act
    sim.draw()
    # Assert
    assert sim.viewer is None
    assert calls == []


# ------------------------------------------------------------------------ publish_truth

def run_truth(people, sight=True, origin=ORIGIN):
    sim = FakeSim(FakeWorld(sight=sight))
    sim.publish_truth(people, origin, EYE, sim.get_clock().now())
    return sim, json.loads(sim.truth[0].data)


def visible(people, **kw):
    return [p['visible'] for p in run_truth(people, **kw)[1]['people']]


def test_truth_message_has_stamp_and_rounded_positions():
    # Arrange / Act
    _, msg = run_truth([(3.14159, -2.71828, 0.0)])
    # Assert
    assert msg['stamp'] == pytest.approx(7.5)
    assert msg['people'] == [dict(person=0, x=3.14, y=-2.72, visible=True)]


def test_truth_skips_people_who_are_not_in_the_scene_but_keeps_their_numbers():
    # Arrange / Act
    _, msg = run_truth([None, (3.0, 0.0, 0.0), None, (4.0, 0.0, 0.0)])
    # Assert
    assert [p['person'] for p in msg['people']] == [1, 3]


def test_truth_person_in_front_within_range_is_visible():
    # Arrange / Act / Assert
    assert visible([(3.0, 0.0, 0.0)]) == [True]


@pytest.mark.parametrize('x, seen', [(RADAR_RANGE - 0.01, True), (RADAR_RANGE, False)])
def test_truth_range_limit_is_strict(x, seen):
    # Arrange / Act / Assert
    assert visible([(x, 0.0, 0.0)]) == [seen]


@pytest.mark.parametrize('angle, seen', [(59.0, True), (61.0, False), (-59.0, True),
                                         (-61.0, False)])
def test_truth_field_of_view_is_plus_minus_sixty_degrees(angle, seen):
    # Arrange: 5 m away at the given angle
    a = math.radians(angle)
    # Act / Assert
    assert visible([(5 * math.cos(a), 5 * math.sin(a), 0.0)]) == [seen]


def test_truth_person_behind_the_radar_is_not_visible():
    # Arrange / Act / Assert
    assert visible([(-3.0, 0.0, 0.0)]) == [False]


def test_truth_person_behind_a_wall_is_not_visible():
    # Arrange / Act
    assert visible([(3.0, 0.0, 0.0)], sight=False) == [False]


def test_truth_line_of_sight_is_checked_at_chest_height():
    # Arrange / Act
    sim, _ = run_truth([(3.0, 1.0, 0.0)])
    # Assert
    assert sim.world.sight_calls == [(3.0, 1.0, 1.0)]


def test_truth_line_of_sight_is_not_asked_for_people_out_of_range():
    # Arrange / Act
    sim, _ = run_truth([(-3.0, 0.0, 0.0)])
    # Assert: the cheap range and angle checks come first
    assert sim.world.sight_calls == []


def test_truth_uses_the_radar_pose_not_the_map_axes():
    # Arrange: radar turned to look along +y; person at +y is in front, at +x is not
    sim = FakeSim()
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # Act
    sim.publish_truth([(0.0, 3.0, 0.0), (3.0, 0.0, 0.0)], np.zeros(3), rot, sim.get_clock().now())
    # Assert
    assert [p['visible'] for p in json.loads(sim.truth[0].data)['people']] == [True, False]


def test_truth_markers_start_with_delete_all_then_one_cylinder_per_person():
    # Arrange / Act
    sim, _ = run_truth([(3.0, 1.0, 0.0), None, (4.0, -1.0, 0.0)])
    # Assert
    markers = sim.markers[0].markers
    assert markers[0].action == Marker.DELETEALL
    people = markers[1:]
    assert [m.id for m in people] == [0, 2]
    assert all(m.ns == 'sim_people' and m.type == Marker.CYLINDER for m in people)
    assert (people[0].pose.position.x, people[0].pose.position.y) == (3.0, 1.0)
    assert people[0].pose.position.z == pytest.approx(0.85)


def test_truth_with_nobody_still_clears_the_markers():
    # Arrange / Act
    sim, msg = run_truth([None, None, None])
    # Assert
    assert msg['people'] == []
    assert len(sim.markers[0].markers) == 1


# ---------------------------------------------------------------------------------- step

def step_sim():
    sim = with_tf(FakeSim(), base_link=(0.0, 0.0, 0.5, 0.0), radar_link=(0.5, 0.0, 0.6, 0.0))
    sim.t0 = __import__('time').monotonic() - 10.0   # scene time of 10 s, person 0 is walking
    return sim


def test_step_waits_when_the_robot_pose_is_missing():
    # Arrange
    sim = FakeSim()
    # Act
    sim.step()
    # Assert
    assert sim.points == [] and sim.truth == [] and sim.world.calls == []


def test_step_waits_when_only_the_base_pose_is_known():
    # Arrange
    sim = with_tf(FakeSim(), base_link=(0.0, 0.0, 0.5, 0.0))
    # Act
    sim.step()
    # Assert
    assert sim.points == [] and sim.world.calls == []


def test_step_scans_and_publishes_points_truth_and_markers():
    # Arrange
    sim = step_sim()
    # Act
    sim.step()
    # Assert: one cloud, stamped with the pose time and holding the scanned points
    (cloud,) = sim.points
    assert cloud.header.stamp.sec == 5
    assert cloud.header.frame_id == 'radar_link'
    assert read_cloud_full(cloud).shape == (2, 5)
    assert len(sim.truth) == 1 and len(sim.markers) == 1
    assert [c[0] for c in sim.world.calls] == ['people', 'robot', 'forward', 'scan']


def test_step_scans_from_the_radar_position_with_zero_speed_on_the_first_frame():
    # Arrange
    sim = step_sim()
    # Act
    sim.step()
    # Assert
    scan = sim.world.calls[-1]
    assert scan[1] == pytest.approx([0.5, 0.0, 0.6])
    assert scan[3] == pytest.approx([0.0, 0.0, 0.0])


def test_step_sensor_speed_is_the_movement_since_the_last_frame_times_the_rate():
    # Arrange: first frame, then the radar moves 0.05 m along x
    sim = step_sim()
    sim.step()
    with_tf(sim, base_link=(0.05, 0.0, 0.5, 0.0), radar_link=(0.55, 0.0, 0.6, 0.0))
    # Act
    sim.step()
    # Assert
    assert sim.world.calls[-1][3] == pytest.approx([0.05 * RADAR_HZ, 0.0, 0.0])
