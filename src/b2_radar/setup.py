from glob import glob
from setuptools import setup

package_name = 'b2_radar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='maintainer@example.com',
    description='Unitree B2 in unitree_mujoco bridged to ROS 2, recorded radar playback',
    license='MIT',
    entry_points={
        'console_scripts': [
            'b2_bridge = b2_radar.b2_bridge_node:main',
            'bag_radar = b2_radar.bag_radar_node:main',
        ],
    },
)
