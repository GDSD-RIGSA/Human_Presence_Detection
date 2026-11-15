"""B2 bridge: quaternion maths, TF / joint publishing and status, without the robot or DDS.

The node methods are borrowed onto a small fake, so neither a ROS graph nor the unitree_go
message package is needed.
"""
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('tf2_ros')
from b2_radar.b2_bridge_node import (IMU_IN_BASE, B2Bridge, _quat_mul,  # noqa: E402
                                     quat_wxyz_to_matrix)
from b2_radar.common import JOINTS, make_tf, spin as SPIN, yaw_quat  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402


class FakeLogger:
    def __init__(self):
        self.lines = []

    def warn(self, text, **kw):
        self.lines.append(('warn', text))

    def info(self, text, **kw):
        self.lines.append(('info', text))


class FakeBridge:
    on_sport = B2Bridge.on_sport
    on_low = B2Bridge.on_low
    check = B2Bridge.check
    publish = B2Bridge.publish
    publish_status = B2Bridge.publish_status

    def __init__(self, post=(0.0, 0.0, 0.0), robot_name='test B2'):
        self.post = list(post)
        self.pos = self.quat = self.q = None
        self.last = 0.0
        self.log = FakeLogger()
        self.sent_tf, self.joints, self.status = [], [], []
        self.tf = SimpleNamespace(sendTransform=self.sent_tf.append)
        self.joint_pub = SimpleNamespace(publish=self.joints.append)
        self.status_pub = SimpleNamespace(publish=self.status.append)
        self._robot_name = robot_name

    def get_logger(self):
        return self.log

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))

    def get_parameter(self, name):
        return SimpleNamespace(value=self._robot_name)

    def ready(self, pos, quat, q=None):
        self.pos, self.quat, self.q = np.array(pos, dtype=float), list(quat), q
        return self


def tfs(bridge):
    """The two transforms of the last publish: (map -> footprint, footprint -> base_link)."""
    (pair,) = bridge.sent_tf
    return pair


def wxyz_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def wxyz_pitch(pitch):
    return (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)


# ------------------------------------------------------------------- quaternion maths

def test_quat_matrix_identity_is_the_identity_matrix():
    # Arrange / Act
    R = quat_wxyz_to_matrix(1.0, 0.0, 0.0, 0.0)
    # Assert
    assert R == pytest.approx(np.eye(3))


def test_quat_matrix_quarter_turn_about_z_maps_x_to_y():
    # Arrange / Act
    R = quat_wxyz_to_matrix(*wxyz_yaw(math.pi / 2))
    # Assert
    assert R @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_quat_matrix_is_a_rotation():
    # Arrange
    q = np.array([0.5, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    # Act
    R = quat_wxyz_to_matrix(*q)
    # Assert
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_quat_mul_identity_leaves_a_quaternion_unchanged():
    # Arrange
    q = (0.5, 0.5, -0.5, 0.5)
    # Act / Assert
    assert _quat_mul((1, 0, 0, 0), q) == pytest.approx(q)
    assert _quat_mul(q, (1, 0, 0, 0)) == pytest.approx(q)


def test_quat_mul_is_not_commutative():
    # Arrange: i * j = k, j * i = -k
    i, j = (0, 1, 0, 0), (0, 0, 1, 0)
    # Act / Assert
    assert _quat_mul(i, j) == pytest.approx((0, 0, 0, 1))
    assert _quat_mul(j, i) == pytest.approx((0, 0, 0, -1))


def test_quat_mul_with_the_conjugate_gives_identity():
    # Arrange
    w, x, y, z = wxyz_yaw(0.7)
    # Act
    prod = _quat_mul((w, -x, -y, -z), (w, x, y, z))
    # Assert
    assert prod == pytest.approx((1, 0, 0, 0))


# ------------------------------------------------------------------ common.py helpers

def test_joints_are_twelve_urdf_names_in_leg_order():
    # Arrange / Act / Assert
    assert len(JOINTS) == 12
    assert JOINTS[:3] == ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint']
    assert JOINTS[-1] == 'RL_calf_joint'
    assert len(set(JOINTS)) == 12


def test_make_tf_sets_frames_translation_and_orientation():
    # Arrange / Act: integer coordinates become floats
    tf = make_tf(Time(sec=3), 'map', 'base_footprint', (1, 2, 3), yaw_quat(math.pi))
    # Assert
    assert (tf.header.frame_id, tf.child_frame_id) == ('map', 'base_footprint')
    assert tf.header.stamp.sec == 3
    t = tf.transform.translation
    assert (t.x, t.y, t.z) == (1.0, 2.0, 3.0)
    assert tf.transform.rotation.z == pytest.approx(1.0)


def test_make_tf_default_orientation_is_identity():
    # Arrange / Act
    q = make_tf(Time(), 'a', 'b', (0, 0, 0)).transform.rotation
    # Assert
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)


def test_yaw_quat_zero_and_half_turn():
    # Arrange / Act / Assert
    assert yaw_quat(0.0) == (0.0, 0.0, 0.0, 1.0)
    assert yaw_quat(math.pi) == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)


# --------------------------------------------------------------------------- callbacks

def test_on_sport_stores_the_position_and_the_time():
    # Arrange
    node = FakeBridge()
    before = time.monotonic()
    # Act
    node.on_sport(SimpleNamespace(position=[1.0, 2.0, 0.5]))
    # Assert
    assert node.pos.tolist() == [1.0, 2.0, 0.5]
    assert node.last >= before


def test_on_low_stores_quaternion_and_the_twelve_motor_angles():
    # Arrange
    node = FakeBridge()
    motors = [SimpleNamespace(q=0.1 * i) for i in range(20)]   # the robot reports more than 12
    msg = SimpleNamespace(imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0]),
                          motor_state=motors)
    # Act
    node.on_low(msg)
    # Assert
    assert node.quat == [1.0, 0.0, 0.0, 0.0]
    assert node.q == pytest.approx([0.1 * i for i in range(12)])


def test_check_warns_while_no_position_has_arrived():
    # Arrange
    node = FakeBridge()
    # Act
    node.check()
    # Assert
    assert node.log.lines[0][0] == 'warn'
    assert '/sportmodestate' in node.log.lines[0][1]


def test_check_is_quiet_once_data_arrives():
    # Arrange
    node = FakeBridge().ready((0, 0, 0.5), (1, 0, 0, 0))
    # Act
    node.check()
    # Assert
    assert node.log.lines == []


# ---------------------------------------------------------------------------- publish

@pytest.mark.parametrize('pos, quat', [(None, (1, 0, 0, 0)), ((0, 0, 0.5), None)])
def test_publish_waits_until_position_and_orientation_are_known(pos, quat):
    # Arrange
    node = FakeBridge()
    node.pos = None if pos is None else np.array(pos, dtype=float)
    node.quat = None if quat is None else list(quat)
    # Act
    node.publish()
    # Assert
    assert node.sent_tf == [] and node.joints == []


def test_publish_level_robot_subtracts_the_imu_offset():
    # Arrange: IMU at (1, 2, 0.5), body level and facing +x
    node = FakeBridge().ready((1.0, 2.0, 0.5), (1, 0, 0, 0))
    # Act
    node.publish()
    # Assert: base_link is the IMU position minus the IMU's offset in the body
    foot, body = tfs(node)
    assert (foot.header.frame_id, foot.child_frame_id) == ('map', 'base_footprint')
    assert (body.header.frame_id, body.child_frame_id) == ('base_footprint', 'base_link')
    assert foot.transform.translation.x == pytest.approx(1.0 - IMU_IN_BASE[0])
    assert foot.transform.translation.y == pytest.approx(2.0 - IMU_IN_BASE[1])
    assert foot.transform.translation.z == 0.0
    assert body.transform.translation.z == pytest.approx(0.5 - IMU_IN_BASE[2])
    assert body.transform.rotation.w == pytest.approx(1.0)


def test_publish_applies_the_simulator_to_map_post():
    # Arrange: world origin placed at (10, 5) in the map, turned 90 degrees
    node = FakeBridge(post=(10.0, 5.0, math.pi / 2)).ready((1.0, 0.0, 0.5), (1, 0, 0, 0))
    base = np.array([1.0, 0.0]) - IMU_IN_BASE[:2]
    # Act
    node.publish()
    # Assert: the body position is rotated by the post yaw, then shifted
    foot, _ = tfs(node)
    assert foot.transform.translation.x == pytest.approx(10.0 - base[1])
    assert foot.transform.translation.y == pytest.approx(5.0 + base[0])
    assert foot.transform.rotation.z == pytest.approx(math.sin(math.pi / 4))
    assert foot.transform.rotation.w == pytest.approx(math.cos(math.pi / 4))


def test_publish_heading_goes_to_the_footprint_and_not_to_base_link():
    # Arrange: robot yawed 90 degrees, no roll or pitch
    node = FakeBridge().ready((0.0, 0.0, 0.5), wxyz_yaw(math.pi / 2))
    # Act
    node.publish()
    # Assert
    foot, body = tfs(node)
    assert foot.transform.rotation.z == pytest.approx(math.sin(math.pi / 4))
    r = body.transform.rotation
    assert (r.x, r.y, r.z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert r.w == pytest.approx(1.0)


def test_publish_pitch_goes_to_base_link_and_not_to_the_footprint():
    # Arrange: body pitched 0.3 rad, facing +x
    node = FakeBridge().ready((0.0, 0.0, 0.5), wxyz_pitch(0.3))
    # Act
    node.publish()
    # Assert
    foot, body = tfs(node)
    assert foot.transform.rotation.y == pytest.approx(0.0, abs=1e-9)
    assert foot.transform.rotation.w == pytest.approx(1.0)
    assert body.transform.rotation.y == pytest.approx(math.sin(0.15))
    assert body.transform.rotation.w == pytest.approx(math.cos(0.15))


def test_publish_joint_states_only_once_the_motor_angles_are_known():
    # Arrange
    node = FakeBridge().ready((0.0, 0.0, 0.5), (1, 0, 0, 0))
    # Act
    node.publish()
    node.q = [0.01 * i for i in range(12)]
    node.publish()
    # Assert: two transforms every time, one joint message after the second call
    assert len(node.sent_tf) == 2
    (js,) = node.joints
    assert js.name == JOINTS
    assert list(js.position) == node.q


# ----------------------------------------------------------------------- publish_status

def test_status_reports_the_robot_name_and_height_while_data_is_fresh():
    # Arrange
    node = FakeBridge(robot_name='B2 sim').ready((0, 0, 0.5678), (1, 0, 0, 0))
    node.last = time.monotonic() - 1.0
    # Act
    node.publish_status()
    # Assert: height is rounded to centimetres
    assert json.loads(node.status[0].data) == dict(robot='B2 sim', height_m=0.57)


def test_status_says_no_data_when_the_robot_went_quiet():
    # Arrange: last position 3 s ago, past the 2 s limit
    node = FakeBridge(robot_name='B2 sim').ready((0, 0, 0.5), (1, 0, 0, 0))
    node.last = time.monotonic() - 3.0
    # Act
    node.publish_status()
    # Assert
    assert json.loads(node.status[0].data)['robot'] == 'no data'


def test_status_before_any_data_has_no_height():
    # Arrange
    node = FakeBridge()
    # Act
    node.publish_status()
    # Assert
    assert json.loads(node.status[0].data) == dict(robot='no data', height_m=None)


# ------------------------------------------------------------------------ construction

def test_node_exits_with_a_hint_when_unitree_messages_are_missing():
    # Arrange: skip on a machine that does have unitree_ros2 sourced
    try:
        import unitree_go.msg  # noqa: F401
        pytest.skip('unitree_go is installed')
    except ImportError:
        pass
    import rclpy
    rclpy.init()
    try:
        # Act / Assert
        with pytest.raises(SystemExit, match='unitree_go messages not found'):
            B2Bridge()
    finally:
        rclpy.try_shutdown()


# ------------------------------------------------------------------------------- spin

class SpinProbe:
    """Replaces rclpy.spin / try_shutdown / ok and signal.signal, and records what spin did."""

    def __init__(self, monkeypatch, spin_raises=None, ok=True):
        import signal

        import rclpy
        self.calls = []
        self.node = SimpleNamespace(destroy_node=lambda: self.calls.append('destroy'))

        def fake_spin(node):
            self.calls.append('spin')
            if spin_raises is not None:
                raise spin_raises

        monkeypatch.setattr(rclpy, 'spin', fake_spin)
        monkeypatch.setattr(rclpy, 'ok', lambda: ok)
        monkeypatch.setattr(rclpy, 'try_shutdown', lambda: self.calls.append('shutdown'))
        monkeypatch.setattr(signal, 'signal', lambda sig, h: self.calls.append(('signal', h)))
        self.sig_ign = signal.SIG_IGN

    def cleanup(self):
        self.calls.append('cleanup')


def test_spin_cleans_up_in_order_after_ctrl_c(monkeypatch):
    # Arrange
    probe = SpinProbe(monkeypatch, spin_raises=KeyboardInterrupt())
    # Act
    SPIN(probe.node, probe.cleanup)
    # Assert: SIGINT is ignored first, so a second Ctrl+C cannot interrupt the cleanup
    assert probe.calls == ['spin', ('signal', probe.sig_ign), 'cleanup', 'destroy', 'shutdown']


def test_spin_without_cleanup_still_destroys_the_node(monkeypatch):
    # Arrange
    probe = SpinProbe(monkeypatch, spin_raises=KeyboardInterrupt())
    # Act
    SPIN(probe.node)
    # Assert
    assert 'cleanup' not in probe.calls
    assert probe.calls[-2:] == ['destroy', 'shutdown']


def test_spin_treats_an_external_shutdown_as_a_normal_exit(monkeypatch):
    # Arrange
    from rclpy.executors import ExternalShutdownException
    probe = SpinProbe(monkeypatch, spin_raises=ExternalShutdownException())
    # Act
    SPIN(probe.node, probe.cleanup)
    # Assert
    assert probe.calls[-3:] == ['cleanup', 'destroy', 'shutdown']


def test_spin_ignores_a_runtime_error_while_shutting_down(monkeypatch):
    # Arrange: rclpy is already not ok, so the error is a message taken during Ctrl+C
    probe = SpinProbe(monkeypatch, spin_raises=RuntimeError('context shut down'), ok=False)
    # Act
    SPIN(probe.node, probe.cleanup)
    # Assert
    assert probe.calls[-3:] == ['cleanup', 'destroy', 'shutdown']


def test_spin_reraises_a_runtime_error_when_rclpy_is_still_running(monkeypatch):
    # Arrange
    probe = SpinProbe(monkeypatch, spin_raises=RuntimeError('real problem'), ok=True)
    # Act / Assert: the error is not hidden, and the cleanup still runs
    with pytest.raises(RuntimeError, match='real problem'):
        SPIN(probe.node, probe.cleanup)
    assert probe.calls[-3:] == ['cleanup', 'destroy', 'shutdown']
