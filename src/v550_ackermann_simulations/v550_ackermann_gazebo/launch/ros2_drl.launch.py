import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

DRL_MODEL = "wheeltec_v550_ackermann"


def generate_launch_description():
    v550_ackermann_description_dir = get_package_share_directory("v550_ackermann_description")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    pause = LaunchConfiguration("pause", default="false")
    gui = LaunchConfiguration("gui", default="false")
    urdf_file = LaunchConfiguration("urdf_file")
    default_urdf_file = os.path.join(v550_ackermann_description_dir, "urdf", DRL_MODEL, "rl_training.urdf")
    world_file_name = LaunchConfiguration("world", default="v550_drl/" + DRL_MODEL + ".model")
    world = PathJoinSubstitution([get_package_share_directory("v550_ackermann_gazebo"), "worlds", world_file_name])
    launch_file_dir = os.path.join(get_package_share_directory("v550_ackermann_gazebo"), "launch")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Launch Gazebo client GUI if true",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="v550_drl/" + DRL_MODEL + ".model",
                description="DRL training world file under v550_ackermann_gazebo/worlds",
            ),
            DeclareLaunchArgument(
                "urdf_file",
                default_value=default_urdf_file,
                description="Robot URDF used by robot_state_publisher during DRL training",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")),
                launch_arguments={"world": world, "pause": pause}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")),
                condition=IfCondition(gui),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([launch_file_dir, "/robot_state_publisher.launch.py"]),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "urdf_file": urdf_file,
                }.items(),
            ),
        ]
    )
