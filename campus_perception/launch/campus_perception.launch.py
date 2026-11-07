"""Independent perception and optional RViz; fusion follows YAML unless overridden."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _make_nodes(context):
    parameters=[LaunchConfiguration('config'), {
        'use_sim_time':ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool)}]
    fusion=LaunchConfiguration('two_frame_fusion').perform(context).strip().lower()
    if fusion in ('true','false'):
        parameters.append({'two_frame_fusion_enabled':fusion=='true'})
    elif fusion!='config':
        raise ValueError('two_frame_fusion must be config, true, or false')
    return [
        Node(package='campus_perception',executable='perception_tracking_node',name='perception_tracking_node',
             parameters=parameters,output='screen'),
        Node(package='rviz2',executable='rviz2',name='campus_perception_rviz',
             condition=IfCondition(LaunchConfiguration('with_rviz')),
             arguments=['-d',LaunchConfiguration('rviz_config')],
             parameters=[{'use_sim_time':ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool)}],output='screen'),
    ]


def generate_launch_description():
    share=Path(get_package_share_directory('campus_perception'))
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value=str(share/'config/perception_tracking.yaml')),
        DeclareLaunchArgument('two_frame_fusion',default_value='config',description='config/true/false'),
        DeclareLaunchArgument('with_rviz',default_value='false'),
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        DeclareLaunchArgument('rviz_config',default_value=str(share/'rviz/perception_tracking.rviz')),
        OpaqueFunction(function=_make_nodes),
    ])
