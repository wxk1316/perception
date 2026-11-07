"""Run the replacement perception only; no planner, controller or UDP bridge."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('campus_perception'))
    config = str(share/'config'/'perception_tracking.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value=config),
        DeclareLaunchArgument('with_rviz',default_value='false'),
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        DeclareLaunchArgument('rviz_config',default_value=str(share/'rviz'/'perception_tracking.rviz')),
        Node(package='campus_perception',executable='perception_tracking_node',name='perception_tracking_node',
             parameters=[LaunchConfiguration('config'), {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],output='screen'),
        Node(package='rviz2',executable='rviz2',name='campus_perception_rviz',
             condition=IfCondition(LaunchConfiguration('with_rviz')),
             arguments=['-d', LaunchConfiguration('rviz_config')],
             parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],output='screen'),
    ])
