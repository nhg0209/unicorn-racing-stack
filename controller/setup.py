from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HMCL',
    maintainer_email='jeongsangryu@gmail.com',
    description='Lateral and longitudinal controllers for F1TENTH autonomous racing',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'controller_manager = controller.controller_manager:main',
            'pp_heading_controller = controller.pp_heading_controller:main',
        ],
    },
)
