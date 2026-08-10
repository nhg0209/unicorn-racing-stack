from setuptools import setup

package_name = 'rate_check'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jonathan Becker',
    maintainer_email='ryujs@unist.ac.kr',
    description='rate_check velocity-profile library (calc_vel_profile)',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [],
    },
)
