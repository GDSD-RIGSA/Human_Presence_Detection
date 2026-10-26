"""Helpers shared by the RIGSA alpha nodes: map loading, PointCloud2 packing/unpacking, poses."""
import math
import os

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

RADAR_FRAME = 'radar_link'
# Frames: map -> base_footprint (robot on the floor) -> base_link (B2 body, URDF root) -> radar_link
RADAR_MOUNT = (0.52, 0.0, 0.06)   # radar_link in base_link, metres: on the B2's nose, facing forward

# Fields published on /radar/points, same names the TI mmWave ROS driver uses (+ radial velocity)
CLOUD_FIELDS = ('x', 'y', 'z', 'intensity', 'velocity')


def load_map(yaml_path):
    """Reads a nav2_map_server style map (.yaml + .pgm). Returns w, h, resolution, origin, grid (row 0 = bottom)."""
    meta = {}
    with open(yaml_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    pgm = os.path.join(os.path.dirname(yaml_path), meta['image'])
    with open(pgm, 'rb') as f:
        data = f.read()
    parts, i = [], 0
    while len(parts) < 4:   # P5 header: magic, width, height, maxval
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
    w, h = int(parts[1]), int(parts[2])
    px = np.frombuffer(data[i + 1:i + 1 + w * h], dtype=np.uint8).reshape(h, w)
    occ = (255.0 - px) / 255.0
    grid = np.full((h, w), -1, dtype=np.int8)
    grid[occ > float(meta['occupied_thresh'])] = 100
    grid[occ < float(meta['free_thresh'])] = 0
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')]
    return w, h, float(meta['resolution']), origin, np.flipud(grid)


def make_cloud(points, stamp, frame=RADAR_FRAME):
    """points: (N, 5) float array with columns CLOUD_FIELDS."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, len(CLOUD_FIELDS))
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.height, msg.width = 1, len(pts)
    msg.fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                  for k, n in enumerate(CLOUD_FIELDS)]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 4 * len(CLOUD_FIELDS), True
    msg.row_step = msg.point_step * len(pts)
    msg.data = pts.tobytes()
    return msg


def read_cloud(msg):
    """Returns an (N, 3) xyz array from any float32 PointCloud2 (e.g. ours or the TI driver's)."""
    offs = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    if n == 0 or not all(k in offs for k in 'xyz'):
        return np.zeros((0, 3), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    cols = [raw[:, offs[k]:offs[k] + 4].copy().view(np.float32).ravel() for k in 'xyz']
    xyz = np.stack(cols, axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def read_cloud_full(msg):
    """Returns an (N, 5) array with columns CLOUD_FIELDS; missing or non-finite intensity/velocity become 0."""
    offs = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    if n == 0 or not all(k in offs for k in 'xyz'):
        return np.zeros((0, len(CLOUD_FIELDS)), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    out = np.zeros((n, len(CLOUD_FIELDS)), dtype=np.float32)
    for k, name in enumerate(CLOUD_FIELDS):
        if name in offs:
            out[:, k] = raw[:, offs[name]:offs[name] + 4].copy().view(np.float32).ravel()
    out = out[np.isfinite(out[:, :3]).all(axis=1)]
    out[~np.isfinite(out)] = 0.0
    return out


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_matrix(tf):
    """geometry_msgs/Transform -> 4x4 homogeneous matrix."""
    q, t = tf.rotation, tf.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
        [0, 0, 0, 1]])


def spin(node, cleanup=None):
    """Spins until Ctrl+C, then shuts down quietly (ros2 launch sends a second SIGINT while we clean up)."""
    import signal

    import rclpy
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:   # a message taken while Ctrl+C shuts the context down; anything else is a real error
        if rclpy.ok():
            raise
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if cleanup is not None:
            cleanup()
        node.destroy_node()
        rclpy.try_shutdown()


def make_tf(stamp, parent, child, xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    from geometry_msgs.msg import TransformStamped
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = [float(v) for v in xyz]
    q = t.transform.rotation
    q.x, q.y, q.z, q.w = [float(v) for v in quat_xyzw]
    return t


def yaw_quat(yaw):
    """(x, y, z, w) quaternion for a rotation about z."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
