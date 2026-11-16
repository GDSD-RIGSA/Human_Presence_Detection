"""Skip test files whose optional dependencies are not installed.

A module-level pytest.importorskip makes pytest 6 (Ubuntu 22.04 / ROS Humble) collect no tests at
all for the whole run, so files that need MuJoCo or Qt are excluded here instead.
"""
import importlib.util


def _missing(module):
    return importlib.util.find_spec(module) is None


collect_ignore = []
if _missing('mujoco'):   # pip install mujoco
    collect_ignore += ['test_mujoco_world.py', 'test_radar_sim_node.py']
if _missing('python_qt_binding'):   # apt install ros-humble-python-qt-binding
    collect_ignore += ['test_operator_panel.py']
