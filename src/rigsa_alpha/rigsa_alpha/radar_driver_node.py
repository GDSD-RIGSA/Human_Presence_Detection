"""Driver for a real TI AWR1443BOOST running the mmWave SDK 2.x out-of-box demo (xWR14xx).

Sends a .cfg (saved from TI's mmWave Demo Visualizer) to the CLI port, then reads the binary frames from the
data port and publishes the detected points like radar_sim does.

Parameters: cli_port (/dev/ttyACM0, 115200), data_port (/dev/ttyACM1, 921600), radar_cfg (path to the .cfg),
            clutter_removal (true: forces `clutterRemoval 1`, so only moving things are reported)
Publishes:  /radar/points     sensor_msgs/PointCloud2 in radar_link: x, y, z, intensity (dB), velocity (m/s)
            /operator/status  std_msgs/String (JSON) for the operator panel footer

TI's radar frame is x right, y forward (boresight), z up; radar_link is x forward, y left, z up.
"""
import json
import math
import struct
import threading
import time

import numpy as np
import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from rigsa_alpha.common import RADAR_FRAME, make_cloud, spin

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
TLV_DETECTED_POINTS = 1
C = 299792458.0


class RadarDriver(Node):
    def __init__(self):
        super().__init__('rigsa_radar_driver')
        self.declare_parameter('cli_port', '/dev/ttyACM0')
        self.declare_parameter('data_port', '/dev/ttyACM1')
        self.declare_parameter('radar_cfg', '')
        self.declare_parameter('clutter_removal', True)
        p = lambda k: self.get_parameter(k).value   # noqa: E731
        self.pub = self.create_publisher(String, '/operator/status', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/radar/points', 10)   # reliable, like radar_sim: works for every subscriber
        self.frames = 0
        self.points = 0
        self.vel_res = 0.0

        lines = self.read_cfg(p('radar_cfg'), p('clutter_removal'))
        self.cli = serial.Serial(p('cli_port'), 115200, timeout=0.5)
        self.data = serial.Serial(p('data_port'), 921600, timeout=0.1)
        self.configure(lines)
        self.running = True
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()
        self.create_timer(1.0, self.publish_status)
        self.create_timer(10.0, self.report)

    def read_cfg(self, path, clutter):
        if not path:
            raise RuntimeError('radar_cfg is not set (path to the .cfg saved from the mmWave Demo Visualizer)')
        lines = []
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('%'):
                    continue
                if line.startswith('clutterRemoval'):
                    line = 'clutterRemoval %d' % (1 if clutter else 0)
                lines.append(line)
        self.doppler_resolution(lines)
        self.get_logger().info('Radar cfg %s: %d commands, clutter removal %s' % (path, len(lines), 'on' if clutter else 'off'))
        return lines

    def doppler_resolution(self, lines):
        """m/s per Doppler bin, from profileCfg and frameCfg."""
        prof = frame = None
        for line in lines:
            w = line.split()
            if w[0] == 'profileCfg':
                prof = w
            elif w[0] == 'frameCfg':
                frame = w
        if prof is None or frame is None:
            return
        start_ghz, idle_us, ramp_us = float(prof[2]), float(prof[3]), float(prof[5])
        n_tx = int(frame[2]) - int(frame[1]) + 1
        loops = int(frame[3])
        n_doppler = 2 ** int(math.ceil(math.log2(max(loops, 1))))
        lam = C / (start_ghz * 1e9)
        self.vel_res = lam / (2.0 * n_doppler * n_tx * (idle_us + ramp_us) * 1e-6)

    def configure(self, lines):
        self.cli.reset_input_buffer()
        for line in lines:
            self.cli.write((line + '\n').encode())
            reply = b''
            end = time.monotonic() + (2.0 if line == 'sensorStart' else 1.0)
            while time.monotonic() < end:
                reply += self.cli.read(256)
                if b'Done' in reply or b'Error' in reply or b'not recognized' in reply:
                    break
            text = reply.decode(errors='replace').replace('mmwDemo:/>', '').strip()
            if b'Done' not in reply:
                level = self.get_logger().error if b'Error' in reply else self.get_logger().warn
                level('CLI "%s" -> %s' % (line, text.replace('\n', ' | ') or 'no reply'))
        self.get_logger().info('Radar configured and started')

    def read_loop(self):
        buf = b''
        while self.running and rclpy.ok():
            try:
                buf += self.data.read(4096)
            except serial.SerialException as e:
                self.get_logger().error('Data port error: %s' % e)
                return
            while True:
                i = buf.find(MAGIC)
                if i < 0:
                    buf = buf[-7:]
                    break
                buf = buf[i:]
                if len(buf) < 16:
                    break
                total = struct.unpack_from('<I', buf, 12)[0]
                if total < 36 or total > 65536:
                    buf = buf[8:]
                    continue
                if len(buf) < total:
                    break
                self.parse_frame(buf[:total])
                buf = buf[total:]

    def parse_frame(self, pkt):
        # header: magic(8) version totalPacketLen platform frameNumber timeCpuCycles numDetectedObj numTLVs
        n_tlv = struct.unpack_from('<I', pkt, 32)[0]
        off = 36
        if n_tlv > 10:   # SDK variants with subFrameNumber in the header (xWR16xx)
            n_tlv = struct.unpack_from('<I', pkt, 36)[0]
            off = 40
        pts = np.zeros((0, 5), dtype=np.float32)
        for _ in range(n_tlv):
            if off + 8 > len(pkt):
                break
            ttype, tlen = struct.unpack_from('<II', pkt, off)
            body = off + 8
            if ttype == TLV_DETECTED_POINTS and body + 4 <= len(pkt):
                n, q = struct.unpack_from('<HH', pkt, body)
                pts = self.parse_points(pkt, body + 4, n, q)
            off = body + tlv_payload(tlen, ttype)
        self.frames += 1
        self.points += len(pts)
        self.cloud_pub.publish(make_cloud(pts, self.get_clock().now().to_msg(), frame=RADAR_FRAME))

    def parse_points(self, pkt, start, n, q):
        n = min(n, (len(pkt) - start) // 12)
        if n <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        raw = np.frombuffer(pkt, dtype=np.dtype([('range', '<u2'), ('doppler', '<i2'), ('peak', '<u2'),
                                                 ('x', '<i2'), ('y', '<i2'), ('z', '<i2')]), count=n, offset=start)
        scale = 1.0 / (1 << q)
        out = np.zeros((n, 5), dtype=np.float32)
        out[:, 0] = raw['y'] * scale          # forward
        out[:, 1] = -raw['x'] * scale         # left
        out[:, 2] = raw['z'] * scale          # up
        out[:, 3] = 10.0 * np.log10(raw['peak'].astype(np.float32) + 1.0)
        out[:, 4] = raw['doppler'] * self.vel_res
        return out

    def publish_status(self):
        self.pub.publish(String(data=json.dumps(dict(robot='TI AWR1443 (room test, no robot)'))))

    def report(self):
        self.get_logger().info('Radar: %d frames in the last 10 s, %d points' % (self.frames, self.points))
        self.frames = self.points = 0

    def stop(self):
        self.running = False
        try:
            self.cli.write(b'sensorStop\n')
            time.sleep(0.2)
        except serial.SerialException:
            pass


def tlv_payload(tlen, ttype):
    """SDK 2.x reports the TLV payload length without the 8-byte TLV header."""
    return tlen


def main():
    rclpy.init()
    node = RadarDriver()
    spin(node, cleanup=node.stop)


if __name__ == '__main__':
    main()
