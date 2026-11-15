"""RIG-49: the TI frame parser and the /radar/points message, without a radar or serial port."""
import struct
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('serial')
from builtin_interfaces.msg import Time  # noqa: E402
from rigsa_alpha.common import CLOUD_FIELDS, RADAR_FRAME, make_cloud, read_cloud_full  # noqa: E402
from rigsa_alpha.radar_driver_node import MAGIC, TLV_DETECTED_POINTS, RadarDriver  # noqa: E402


def ti_frame(points, q=7):
    """
    One xWR14xx SDK 2.x frame: 36-byte header + a detected-points TLV.

    points: (range, doppler, peak, x, y, z).
    """
    body = struct.pack('<HH', len(points), q) + b''.join(
        struct.pack('<HhHhhh', *p) for p in points)
    tlv = struct.pack('<II', TLV_DETECTED_POINTS, len(body)) + body
    total = 36 + len(tlv)
    header = MAGIC + struct.pack('<7I', 0x01000000, total, 0x1443, 1, 0, len(points), 1)
    return header + tlv


class FakeDriver:
    """Just enough of RadarDriver for parse_frame: captures the published cloud."""
    parse_points = RadarDriver.parse_points

    def __init__(self, vel_res):
        self.vel_res = vel_res
        self.frames = self.points = 0
        self.sent = []
        self.cloud_pub = SimpleNamespace(publish=self.sent.append)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))


def test_frame_to_radar_points():
    q = 7   # TI Q format: value / 2^q metres
    drv = FakeDriver(vel_res=0.1)
    pkt = ti_frame([(0, -3, 999, 128, 256, 64),    # x=1.0 right, y=2.0 forward, z=0.5 up
                    (0, 5, 0, -64, 384, 0)], q)     # x=0.5 left, y=3.0 forward
    RadarDriver.parse_frame(drv, pkt)
    assert drv.frames == 1 and drv.points == 2
    msg = drv.sent[0]
    assert msg.header.frame_id == RADAR_FRAME == 'radar_link'
    assert [f.name for f in msg.fields] == list(CLOUD_FIELDS) == [
        'x', 'y', 'z', 'intensity', 'velocity']
    pts = read_cloud_full(msg)
    # radar_link: x forward, y left, z up
    np.testing.assert_allclose(pts[:, :3], [[2.0, -1.0, 0.5], [3.0, 0.5, 0.0]], atol=1e-6)
    # intensity, dB
    np.testing.assert_allclose(pts[:, 3], [10 * np.log10(1000.0), 0.0], rtol=1e-6)
    # velocity, m/s
    np.testing.assert_allclose(pts[:, 4], [-0.3, 0.5], atol=1e-6)


def test_empty_frame_publishes_empty_cloud():
    drv = FakeDriver(vel_res=0.1)
    RadarDriver.parse_frame(drv, ti_frame([]))
    assert drv.frames == 1 and drv.sent[0].width == 0


def test_doppler_resolution_from_cfg():
    drv = SimpleNamespace(vel_res=0.0)
    lines = ['profileCfg 0 77 186 7 57.14 0 0 70 1 240 4884 0 0 30', 'frameCfg 0 1 16 0 100 1 0']
    RadarDriver.doppler_resolution(drv, lines)
    # lambda / (2 * N_doppler * N_tx * (idle + ramp)) with 16 loops, 2 TX, 186 + 57.14 us
    lam = 299792458.0 / 77e9
    assert drv.vel_res == pytest.approx(lam / (2 * 16 * 2 * 243.14e-6))


def test_make_cloud_round_trip():
    pts = np.array([[1.0, 2.0, 3.0, 20.0, -0.5]], dtype=np.float32)
    msg = make_cloud(pts, Time())
    assert msg.point_step == 20 and msg.width == 1
    np.testing.assert_array_equal(read_cloud_full(msg), pts)


def test_wrong_tlv_type_is_ignored():
    drv = FakeDriver(vel_res=0.1)
    pkt = bytearray(ti_frame([(0, 0, 10, 128, 256, 0)]))
    pkt[36:40] = struct.pack('<I', 99)   # not TLV_DETECTED_POINTS
    RadarDriver.parse_frame(drv, bytes(pkt))
    assert drv.frames == 1 and drv.points == 0
    assert drv.sent[0].width == 0


def test_truncated_point_list_keeps_only_complete_points():
    drv = FakeDriver(vel_res=0.1)
    pkt = ti_frame([(0, 0, 10, 128, 256, 0), (0, 0, 10, 64, 128, 0)])
    RadarDriver.parse_frame(drv, pkt[:-6])   # cut the second point in half
    assert drv.points == 1
    assert drv.sent[0].width == 1


def test_point_count_larger_than_payload_is_clamped():
    drv = FakeDriver(vel_res=0.1)
    pkt = bytearray(ti_frame([(0, 0, 10, 128, 256, 0)]))
    pkt[44:46] = struct.pack('<H', 500)   # header says 500 points, only 1 is there
    RadarDriver.parse_frame(drv, bytes(pkt))
    assert drv.points == 1


def test_truncated_tlv_header_publishes_empty_cloud():
    drv = FakeDriver(vel_res=0.1)
    pkt = ti_frame([(0, 0, 10, 128, 256, 0)])
    RadarDriver.parse_frame(drv, pkt[:36 + 5])   # header + 5 bytes of the TLV header
    assert drv.frames == 1 and drv.points == 0


def test_doppler_resolution_without_profile_keeps_zero():
    drv = SimpleNamespace(vel_res=0.0)
    RadarDriver.doppler_resolution(drv, ['frameCfg 0 1 16 0 100 1 0'])   # no profileCfg
    assert drv.vel_res == 0.0


@pytest.mark.parametrize('loops, n_doppler', [(0, 1), (1, 1), (3, 4)])
def test_doppler_resolution_rounds_loops_up_to_a_power_of_two(loops, n_doppler):
    drv = SimpleNamespace(vel_res=0.0)
    lines = ['profileCfg 0 77 186 7 57.14 0 0 70 1 240 4884 0 0 30',
             'frameCfg 0 1 %d 0 100 1 0' % loops]
    RadarDriver.doppler_resolution(drv, lines)
    lam = 299792458.0 / 77e9
    assert drv.vel_res == pytest.approx(lam / (2 * n_doppler * 2 * 243.14e-6))


def test_header_with_subframe_field_is_parsed():
    drv = FakeDriver(vel_res=0.1)
    pkt = ti_frame([(0, 0, 10, 128, 256, 0)])
    # xWR16xx variant: the number-of-TLVs slot holds another value, the real count follows it
    variant = pkt[:32] + struct.pack('<II', 99, 1) + pkt[36:]
    RadarDriver.parse_frame(drv, variant)
    assert drv.points == 1


def cfg_driver():
    return SimpleNamespace(doppler_resolution=lambda lines: None,
                           get_logger=lambda: SimpleNamespace(info=lambda msg: None))


def test_read_cfg_without_a_path_raises():
    with pytest.raises(RuntimeError):
        RadarDriver.read_cfg(cfg_driver(), '', True)


@pytest.mark.parametrize('clutter, expected', [(True, 'clutterRemoval 1'),
                                               (False, 'clutterRemoval 0')])
def test_read_cfg_skips_comments_and_forces_clutter_removal(tmp_path, clutter, expected):
    cfg = tmp_path / 'radar.cfg'
    cfg.write_text('% saved from the visualizer\n\nclutterRemoval -1 0\nsensorStart\n')
    lines = RadarDriver.read_cfg(cfg_driver(), str(cfg), clutter)
    assert lines == [expected, 'sensorStart']


def test_empty_cloud_round_trip():
    msg = make_cloud(np.zeros((0, 5)), Time())
    assert msg.width == 0 and len(msg.data) == 0
    assert read_cloud_full(msg).shape == (0, 5)
