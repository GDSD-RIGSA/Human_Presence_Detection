"""Helpers shared by the b2_radar nodes: frames, TF messages, spinning."""
import math

RADAR_FRAME = 'radar_link'
# Frames: map -> base_footprint (robot on the floor) -> base_link (B2 body, URDF root) -> radar_link
RADAR_MOUNT = (0.52, 0.0, 0.06)   # radar_link in base_link, metres: on the B2's nose, facing forward
JOINTS = ['%s_%s_joint' % (leg, j) for leg in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]   # b2_description URDF


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
