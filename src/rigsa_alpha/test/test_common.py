"""Pose maths, map loading and PointCloud2 helpers in rigsa_alpha.common (no running ROS graph)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from builtin_interfaces.msg import Time  # noqa: E402
from rigsa_alpha.common import (CLOUD_FIELDS, load_map, make_cloud, make_tf,  # noqa: E402
                                quat_to_yaw, read_cloud, transform_matrix, yaw_quat)
from rigsa_alpha.common import spin as SPIN  # noqa: E402


def quat(x, y, z, w):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


# ---------------------------------------------------------------- yaw and quaternions

@pytest.mark.parametrize('yaw', [0.0, 0.5, math.pi / 2, -math.pi / 2, 3.0, -3.0])
def test_yaw_quat_round_trips_through_quat_to_yaw(yaw):
    # Arrange / Act
    x, y, z, w = yaw_quat(yaw)
    # Assert
    assert quat_to_yaw(quat(x, y, z, w)) == pytest.approx(yaw)


def test_yaw_quat_is_a_unit_quaternion_about_z():
    # Arrange / Act
    x, y, z, w = yaw_quat(1.234)
    # Assert
    assert (x, y) == (0.0, 0.0)
    assert x * x + y * y + z * z + w * w == pytest.approx(1.0)


def test_quat_to_yaw_identity_is_zero():
    # Arrange / Act / Assert
    assert quat_to_yaw(quat(0.0, 0.0, 0.0, 1.0)) == 0.0


def test_quat_to_yaw_half_turn_is_pi_in_magnitude():
    # Arrange / Act: boundary of the (-pi, pi] range, the sign of pi is not important
    yaw = quat_to_yaw(quat(*yaw_quat(math.pi)))
    # Assert
    assert abs(yaw) == pytest.approx(math.pi)


# ------------------------------------------------------------------- make_tf, matrix

def test_make_tf_sets_frames_translation_and_default_orientation():
    # Arrange
    stamp = Time(sec=12, nanosec=5)
    # Act: integer coordinates must become floats for the message
    tf = make_tf(stamp, 'map', 'base_link', (1, 2, 3))
    # Assert
    assert (tf.header.frame_id, tf.child_frame_id) == ('map', 'base_link')
    assert (tf.header.stamp.sec, tf.header.stamp.nanosec) == (12, 5)
    t = tf.transform.translation
    assert (t.x, t.y, t.z) == (1.0, 2.0, 3.0)
    q = tf.transform.rotation
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)


def test_make_tf_keeps_a_given_orientation():
    # Arrange
    q = yaw_quat(math.pi / 2)
    # Act
    tf = make_tf(Time(), 'a', 'b', (0, 0, 0), q)
    # Assert
    r = tf.transform.rotation
    assert (r.x, r.y, r.z, r.w) == pytest.approx(q)


def test_transform_matrix_identity_with_translation():
    # Arrange
    tf = make_tf(Time(), 'map', 'radar_link', (1.5, -2.0, 0.5)).transform
    # Act
    T = transform_matrix(tf)
    # Assert
    assert T[:3, :3] == pytest.approx(np.eye(3))
    assert T[:3, 3] == pytest.approx([1.5, -2.0, 0.5])
    assert T[3] == pytest.approx([0, 0, 0, 1])


def test_transform_matrix_quarter_turn_about_z_maps_x_to_y():
    # Arrange
    tf = make_tf(Time(), 'map', 'radar_link', (0, 0, 0), yaw_quat(math.pi / 2)).transform
    # Act
    moved = transform_matrix(tf) @ np.array([1.0, 0.0, 0.0, 1.0])
    # Assert
    assert moved[:3] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_transform_matrix_rotation_then_translation_in_that_order():
    # Arrange: point (1, 0) turned 90 deg, then shifted by (10, 0)
    tf = make_tf(Time(), 'map', 'r', (10, 0, 0), yaw_quat(math.pi / 2)).transform
    # Act
    moved = transform_matrix(tf) @ np.array([1.0, 0.0, 0.0, 1.0])
    # Assert
    assert moved[:3] == pytest.approx([10.0, 1.0, 0.0], abs=1e-9)


def test_transform_matrix_rotation_part_is_orthonormal():
    # Arrange
    tf = make_tf(Time(), 'a', 'b', (0, 0, 0), (0.1, 0.2, 0.3, math.sqrt(1 - 0.14))).transform
    # Act
    R = transform_matrix(tf)[:3, :3]
    # Assert
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


# ------------------------------------------------------------------------ read_cloud

def test_read_cloud_returns_the_xyz_columns():
    # Arrange
    pts = np.array([[1, 2, 3, 20, 0.5], [4, 5, 6, 30, -0.5]], dtype=np.float32)
    # Act
    xyz = read_cloud(make_cloud(pts, Time()))
    # Assert
    assert xyz.shape == (2, 3)
    assert xyz == pytest.approx(pts[:, :3])


def test_read_cloud_empty_cloud_gives_no_points():
    # Arrange / Act
    xyz = read_cloud(make_cloud(np.zeros((0, len(CLOUD_FIELDS))), Time()))
    # Assert
    assert xyz.shape == (0, 3)


def test_read_cloud_drops_points_with_non_finite_coordinates():
    # Arrange
    pts = np.array([[1, 2, 3, 0, 0], [np.nan, 2, 3, 0, 0], [1, np.inf, 3, 0, 0]],
                   dtype=np.float32)
    # Act
    xyz = read_cloud(make_cloud(pts, Time()))
    # Assert
    assert xyz == pytest.approx(np.array([[1, 2, 3]]))


def test_read_cloud_without_an_xyz_field_gives_no_points():
    # Arrange
    msg = make_cloud(np.ones((2, len(CLOUD_FIELDS))), Time())
    msg.fields = [f for f in msg.fields if f.name != 'z']
    # Act
    xyz = read_cloud(msg)
    # Assert
    assert xyz.shape == (0, 3)


# ------------------------------------------------------------------------- load_map

def write_map(tmp_path, pixels, width, height, extra=''):
    (tmp_path / 'm.pgm').write_bytes(b'P5\n%d %d\n255\n' % (width, height) + bytes(pixels))
    yaml = tmp_path / 'm.yaml'
    yaml.write_text('image: m.pgm\nresolution: 0.05\norigin: [-1.0, -2.0, 0.0]\n'
                    'occupied_thresh: 0.65\nfree_thresh: 0.196\nnegate: 0\n' + extra)
    return str(yaml)


def test_load_map_reads_size_resolution_and_origin(tmp_path):
    # Arrange
    path = write_map(tmp_path, [255] * 6, 3, 2)
    # Act
    w, h, res, origin, grid = load_map(path)
    # Assert
    assert (w, h, res, origin) == (3, 2, 0.05, [-1.0, -2.0, 0.0])
    assert grid.shape == (2, 3)


def test_load_map_classifies_cells_and_puts_the_bottom_row_first(tmp_path):
    # Arrange: top row black (occupied) + white (free); bottom row grey (unknown) + white
    path = write_map(tmp_path, [0, 254, 205, 255], 2, 2)
    # Act
    grid = load_map(path)[4]
    # Assert: row 0 of the returned grid is the bottom of the image
    assert grid.tolist() == [[-1, 0], [100, 0]]


@pytest.mark.parametrize('pixel, cell', [(0, 100), (89, 100), (90, -1), (205, -1), (206, 0)])
def test_load_map_threshold_boundaries(tmp_path, pixel, cell):
    # Arrange: occupied when (255 - px) / 255 > 0.65, free when < 0.196, otherwise unknown
    path = write_map(tmp_path, [pixel], 1, 1)
    # Act
    grid = load_map(path)[4]
    # Assert
    assert grid[0, 0] == cell


def test_load_map_missing_image_key_raises(tmp_path):
    # Arrange
    yaml = tmp_path / 'bad.yaml'
    yaml.write_text('resolution: 0.05\norigin: [0, 0, 0]\n')
    # Act / Assert
    with pytest.raises(KeyError):
        load_map(str(yaml))


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
