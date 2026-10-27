from glob import glob
from setuptools import setup

package_name = 'rigsa_alpha'

setup(
    name=package_name,
    version='0.3.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/sql', glob('sql/*.sql')),
        ('share/' + package_name + '/params', glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='RIGSA Robotics',
    maintainer_email='rigsa@example.com',
    description='RIGSA alpha demo: Unitree B2 (unitree_mujoco) detects people with simulated mmWave radar, alerts stored in PostgreSQL',
    license='MIT',
    entry_points={
        'console_scripts': [
            'site = rigsa_alpha.site_node:main',
            'scripted_robot = rigsa_alpha.scripted_robot_node:main',
            'radar_sim = rigsa_alpha.radar_sim_node:main',
            'radar_driver = rigsa_alpha.radar_driver_node:main',
            'make_unitree_scene = rigsa_alpha.make_unitree_scene:main',
            'detector = rigsa_alpha.detector_node:main',
            'evaluator = rigsa_alpha.evaluator_node:main',
        ],
    },
)
