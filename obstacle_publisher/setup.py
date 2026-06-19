from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obstacle_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc5',
    maintainer_email='jeongsangryu@gmail.com',
    description='Dynamic and static obstacle publisher for F1TENTH simulator',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'static_obstacle_manager = obstacle_publisher.static_obstacle_manager:main',
            'dynamic_obstacle_publisher = obstacle_publisher.dynamic_obstacle_publisher:main',
        ],
    },
)
