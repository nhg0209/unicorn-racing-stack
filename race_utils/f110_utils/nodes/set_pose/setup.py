from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'set_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc5',
    maintainer_email='shineejoon@unist.ac.kr',
    description='Set cartographer initial pose and save cartographer maps via RViz interactions',
    license='MIT',
    entry_points={
        'console_scripts': [
            'set_pose_node = set_pose.set_pose_node:main',
            'set_pose_v2_node = set_pose.set_pose_v2_node:main',
            'save_carto_map = set_pose.save_carto_map:main',
        ],
    },
)
