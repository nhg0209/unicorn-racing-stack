from setuptools import setup
package_name = 'frenet_conversion'
setup(
    name=package_name, version='0.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Jonathan Becker', maintainer_email='jonbecke@student.ethz.ch',
    description='Frenet<->global conversion library (Python).', license='Apache License 2.0',
    entry_points={'console_scripts': [
        'frenet_converter_demo = frenet_conversion.frenet_converter_demo_node:main',
    ]},
)
