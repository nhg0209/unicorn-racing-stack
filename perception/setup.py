from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc5',
    maintainer_email='jeongsangryu@gmail.com',
    description='LiDAR-based obstacle detection and tracking for F1TENTH autonomous racing',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'detect_node   = perception.detect_ros:main',
            'tracking_node = perception.tracking_ros:main',
        ],
    },
)
