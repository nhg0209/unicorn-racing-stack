import os
from glob import glob
from setuptools import setup

package_name = 'opponent'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='js',
    maintainer_email='kwonc@unist.ac.kr',
    description='Engine-agnostic virtual opponent (VIL) for sim and real car',
    license='MIT',
    entry_points={
        'console_scripts': [
            'opponent_vehicle = opponent.opponent_vehicle:main',
            'opponent_controller = opponent.opponent_controller:main',
            'scan_augmentor = opponent.scan_augmentor:main',
            'static_obstacle_manager = opponent.static_obstacle_manager:main',
            'obstacle_merger = opponent.obstacle_merger:main',
        ],
    },
)
