"""RIGSA alpha demo: B2 + radar simulation, person detector, event log (PostgreSQL), RViz2 and the operator panel.

  robot:=unitree   (default) the B2 in unitree_mujoco, which must already be running:
                   ./unitree_mujoco -r b2 -s scene_rigsa.xml -i 1 -n lo
                   Run this launch in a terminal with `source ~/unitree_ros2/setup_local.sh` and ROS_DOMAIN_ID=1.
                   Also starts b2_radar's b2_bridge (RIG-47) and b2_sim_control's stand node (stand:=false to skip).
  robot:=scripted  no unitree stack needed: a scripted B2 patrols the hallway
  detector_params:=<yaml>  detector overrides, e.g. share/rigsa_alpha/params/room_improved.yaml (room mode)
  robot:=replay    no simulation: play a recorded bag in a second terminal (ros2 bag play <bag>)
  robot:=none      room test with the real TI AWR1443 on a stand (no robot): radar_cfg:=<.cfg from the TI visualizer>,
                   radar_height:=1.0, cli_port:=/dev/ttyACM0, data_port:=/dev/ttyACM1. No evaluator (no ground truth).

event_log / panel: auto (default) starts the event logger and operator panel only when the event log
(rigsa_alpha.event_store) is installed; true / false to force.
Other options: radar_gui:=true (MuJoCo window of the radar world: building, people, B2), rviz:=false, panel:=false,
evaluate:=false, operator:=<name>, seed:=<n>, b2_xml:=<path to unitree_robots/b2/b2.xml>,
db_dsn:="host=localhost port=5432 dbname=rigsa user=rigsa password=rigsa"
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node

from rigsa_alpha.scenario import SENTRY_POST

try:
    from rigsa_alpha.event_store import DEFAULT_DSN
    HAVE_EVENT_LOG = True
except ImportError:   # event log (RIG-56..59) not installed: run the detector without it
    DEFAULT_DSN = 'host=localhost port=5432 dbname=rigsa user=rigsa password=rigsa'
    HAVE_EVENT_LOG = False


# Real radar points are sparser and jumpier than simulated ones (clutter removal, ~15 deg azimuth): keep tracks
# through longer gaps and give the same ID back to a person who reappears nearby.
ROOM_TRACKING = {'gate': 2.0, 'lost_after': 4.0, 'reacquire_s': 15.0, 'reacquire_m': 3.0, 'eps': 0.6}


def b2_urdf():
    try:
        path = os.path.join(get_package_share_directory('b2_description'), 'urdf', 'b2_description.urdf')
    except PackageNotFoundError:
        return None
    with open(path) as f:
        return f.read()


def setup(context):
    a = {k: v for k, v in context.launch_configurations.items()}
    on = lambda k: a[k].lower() in ('true', '1', 'yes') or (a[k] == 'auto' and HAVE_EVENT_LOG)   # noqa: E731
    robot = a['robot']
    if robot not in ('unitree', 'scripted', 'replay', 'none'):
        raise RuntimeError("robot must be 'unitree', 'scripted', 'replay' or 'none', not %r" % robot)
    room = robot == 'none'
    share = get_package_share_directory('rigsa_alpha')
    urdf = None if room else b2_urdf()
    actions = [Node(package='rigsa_alpha', executable='site', output='screen',
                    parameters=[{'robot_box': urdf is None and not room, 'room': room,
                                 'radar_height': float(a['radar_height'])}])]
    if room:
        actions.append(Node(package='rigsa_alpha', executable='radar_driver', output='screen',
                            parameters=[{'radar_cfg': os.path.expanduser(a['radar_cfg']), 'cli_port': a['cli_port'],
                                         'data_port': a['data_port']}]))
    elif urdf is None:
        actions.append(LogInfo(msg='b2_description not found: RViz shows a box instead of the B2 model'))
    else:
        actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
                            parameters=[{'robot_description': urdf}]))
    if robot == 'unitree':
        actions.append(Node(package='b2_radar', executable='b2_bridge', output='screen',
                            parameters=[{'post': [float(v) for v in SENTRY_POST]}]))
        if on('stand'):
            actions.append(Node(package='b2_sim_control', executable='stand', output='screen'))
    elif robot == 'scripted':
        actions.append(Node(package='rigsa_alpha', executable='scripted_robot', output='screen'))
    if robot in ('unitree', 'scripted'):
        actions.append(Node(package='rigsa_alpha', executable='radar_sim', output='screen',
                            parameters=[{'gui': on('radar_gui'), 'b2_xml': os.path.expanduser(a['b2_xml']),
                                         'seed': int(a['seed'])}]))
    actions += [
        Node(package='rigsa_alpha', executable='detector', output='screen',
             parameters=([ROOM_TRACKING] if room else [])
             + ([os.path.expanduser(a['detector_params'])] if a['detector_params'] else [])),
    ]
    if on('event_log'):
        actions.append(Node(package='rigsa_alpha', executable='event_logger', output='screen',
                            parameters=[{'db_dsn': a['db_dsn'], 'source': 'real' if room else robot}]))
    if on('evaluate') and not room:
        actions.append(Node(package='rigsa_alpha', executable='evaluator', output='screen'))
    if on('panel'):
        actions.append(Node(package='rigsa_alpha', executable='operator_panel', output='screen',
                            parameters=[{'db_dsn': a['db_dsn'], 'operator': a['operator']}]))
    if on('rviz'):
        actions.append(Node(package='rviz2', executable='rviz2', output='log',
                            arguments=['-d', os.path.join(share, 'rviz', 'operator.rviz')]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='unitree', description="'unitree', 'scripted' or 'replay'"),
        DeclareLaunchArgument('stand', default_value='true', description='robot:=unitree: start b2_sim_control stand'),
        DeclareLaunchArgument('b2_xml', default_value='~/unitree_mujoco/unitree_robots/b2/b2.xml'),
        DeclareLaunchArgument('radar_gui', default_value='false', description='MuJoCo window of the radar world'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('panel', default_value='auto',
                              description='operator alerts panel (auto: on when the event log is installed)'),
        DeclareLaunchArgument('event_log', default_value='auto',
                              description='event_logger node (auto: on when the event log is installed)'),
        DeclareLaunchArgument('evaluate', default_value='true', description='score detections against ground truth'),
        DeclareLaunchArgument('db_dsn', default_value=DEFAULT_DSN),
        DeclareLaunchArgument('operator', default_value='operator'),
        DeclareLaunchArgument('seed', default_value='-1'),
        DeclareLaunchArgument('radar_cfg', default_value='', description='robot:=none: .cfg for the real radar'),
        DeclareLaunchArgument('radar_height', default_value='1.0', description='robot:=none: radar height, m'),
        DeclareLaunchArgument('cli_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('data_port', default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('detector_params', default_value='',
                              description='yaml with detector overrides, e.g. params/room_improved.yaml'),
        OpaqueFunction(function=setup),
    ])
