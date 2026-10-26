"""RIG-47: the Unitree B2 in unitree_mujoco, bridged to standard ROS 2 topics.

unitree_mujoco must already be running (scripts/run_b2_sim.sh starts it and then this launch):
    ./unitree_mujoco -r b2 -s scene.xml -i 1 -n lo
Run in a terminal with `source ~/unitree_ros2/setup_local.sh`, the B2 workspace sourced and ROS_DOMAIN_ID=1.

Starts b2_bridge (/sportmodestate + /lowstate -> /tf, /joint_states, /operator/status), robot_state_publisher with
the b2_description URDF (if found) and b2_sim_control's stand node (stand:=false to skip).
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node


def b2_urdf():
    try:
        path = os.path.join(get_package_share_directory('b2_description'), 'urdf', 'b2_description.urdf')
    except PackageNotFoundError:
        return None
    with open(path) as f:
        return f.read()


def setup(context):
    a = context.launch_configurations
    actions = [Node(package='b2_radar', executable='b2_bridge', output='screen')]
    urdf = b2_urdf()
    if urdf is None:
        actions.append(LogInfo(msg='b2_description not found: no robot_state_publisher (TF stops at base_link)'))
    else:
        actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
                            parameters=[{'robot_description': urdf}]))
    if a['stand'].lower() in ('true', '1', 'yes'):
        actions.append(Node(package='b2_sim_control', executable='stand', output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('stand', default_value='true', description="start b2_sim_control's stand node"),
        OpaqueFunction(function=setup),
    ])
