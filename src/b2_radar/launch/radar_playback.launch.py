"""RIG-48: recorded radar input on /radar/points, replayed the same way every run.

    ros2 launch b2_radar radar_playback.launch.py bag:=<rosbag2 folder> [loop:=false] [rate:=1.0]

bag_radar publishes only /radar/points (frame radar_link), restamped to the current time, with the recorded frame
spacing. A static TF base_link -> radar_link puts the radar on the B2's nose (radar_tf:=false to skip).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from b2_radar.common import RADAR_FRAME, RADAR_MOUNT


def generate_launch_description():
    x, y, z = (str(v) for v in RADAR_MOUNT)
    return LaunchDescription([
        DeclareLaunchArgument('bag', description='rosbag2 folder with /radar/points'),
        DeclareLaunchArgument('loop', default_value='true'),
        DeclareLaunchArgument('rate', default_value='1.0', description='playback speed'),
        DeclareLaunchArgument('radar_tf', default_value='true', description='publish base_link -> radar_link'),
        Node(package='b2_radar', executable='bag_radar', output='screen',
             parameters=[{'bag': LaunchConfiguration('bag'), 'loop': LaunchConfiguration('loop'),
                          'rate': LaunchConfiguration('rate')}]),
        Node(package='tf2_ros', executable='static_transform_publisher', output='log',
             condition=IfCondition(LaunchConfiguration('radar_tf')),
             arguments=['--x', x, '--y', y, '--z', z, '--frame-id', 'base_link', '--child-frame-id', RADAR_FRAME]),
    ])
