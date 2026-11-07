from setuptools import setup, find_packages
from glob import glob

package_name = 'campus_perception'
setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/models', ['models/THIRD_PARTY_NOTICES.md']),
        ('share/' + package_name + '/models/centerpoint/base', glob('models/centerpoint/base/*')),
        ('share/' + package_name + '/models/centerpoint/tiny', glob('models/centerpoint/tiny/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Campus Perception contributors',
    maintainer_email='dev@campus-perception.local',
    description='Independent raw-LiDAR perception, pretrained CenterPoint and source-time tracking.',
    license='MIT',
    entry_points={'console_scripts': [
        'perception_tracking_node = campus_perception.perception_tracking_node:main',
        'campus_model = campus_perception.centerpoint_model:main',
    ]},
)
