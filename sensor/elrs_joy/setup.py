from glob import glob

from setuptools import find_packages, setup

package_name = "elrs_joy"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="ELRS receiver to sensor_msgs/Joy bridge (ROS 2 Jazzy, CP2102 + FT232 variants)",
    license="MIT",
    entry_points={"console_scripts": [
        "elrs_joy_node = elrs_joy.elrs_joy_node:main",
        "elrs_joy_ft232_node = elrs_joy.elrs_joy_ft232_node:main",
    ]},
)
