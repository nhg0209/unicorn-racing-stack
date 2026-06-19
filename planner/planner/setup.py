from setuptools import find_packages, setup

package_name = 'planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/create_path.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc5',
    maintainer_email='jeongsangryu@gmail.com',
    description='Global planner: centerline extraction and trajectory optimization for F1TENTH',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'centerline_extractor = planner.centerline_extractor:main',
            'trajectory_optimizer = planner.trajectory_optimizer:main',
            'waypoint_publisher = planner.waypoint_publisher:main',
        ],
    },
)
