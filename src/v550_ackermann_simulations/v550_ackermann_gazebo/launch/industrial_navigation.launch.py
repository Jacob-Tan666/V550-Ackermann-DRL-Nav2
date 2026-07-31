import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ROBOT_MODEL = "wheeltec_v550_ackermann"


def generate_launch_description():
    v550_ackermann_gazebo_dir = get_package_share_directory("v550_ackermann_gazebo")
    v550_ackermann_gazebo_prefix = get_package_prefix("v550_ackermann_gazebo")
    v550_ackermann_description_dir = get_package_share_directory("v550_ackermann_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    launch_file_dir = os.path.join(v550_ackermann_gazebo_dir, "launch")
    window_tiler = os.path.join(
        v550_ackermann_gazebo_prefix,
        "lib",
        "v550_ackermann_gazebo",
        "tile_visualization_windows.py",
    )

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    gui = LaunchConfiguration("gui", default="true")
    rviz = LaunchConfiguration("rviz", default="true")
    tile_windows = LaunchConfiguration("tile_windows", default="true")
    pause = LaunchConfiguration("pause", default="false")
    world_file_name = LaunchConfiguration("world", default="warehouse_demo/" + ROBOT_MODEL + ".model")
    dynamic_obstacles = LaunchConfiguration("dynamic_obstacles", default="true")
    dynamic_speed_scale = LaunchConfiguration("dynamic_speed_scale", default="0.8")
    nav2_startup_timeout = LaunchConfiguration("nav2_startup_timeout", default="90.0")
    odom_tf_bridge = LaunchConfiguration("odom_tf_bridge", default="true")
    urdf_file = LaunchConfiguration("urdf_file")
    default_urdf_file = os.path.join(v550_ackermann_description_dir, "urdf", ROBOT_MODEL, "industrial_navigation.urdf")
    params_file = LaunchConfiguration(
        "params_file",
        default=os.path.join(v550_ackermann_gazebo_dir, "config", "industrial_nav2_params.yaml"),
    )
    map_yaml = LaunchConfiguration(
        "map",
        default=os.path.join(v550_ackermann_gazebo_dir, "maps", "industrial_warehouse.yaml"),
    )
    rviz_config = LaunchConfiguration(
        "rviz_config",
        default=os.path.join(v550_ackermann_gazebo_dir, "rviz", "industrial_nav2.rviz"),
    )
    nav_to_pose_bt_xml = LaunchConfiguration(
        "nav_to_pose_bt_xml",
        default=os.path.join(
            v550_ackermann_gazebo_dir,
            "behavior_trees",
            "ackermann_navigate_to_pose_w_replanning_and_recovery.xml",
        ),
    )
    nav_through_poses_bt_xml = LaunchConfiguration(
        "nav_through_poses_bt_xml",
        default=os.path.join(
            v550_ackermann_gazebo_dir,
            "behavior_trees",
            "ackermann_navigate_through_poses_w_replanning_and_recovery.xml",
        ),
    )
    world = PathJoinSubstitution([v550_ackermann_gazebo_dir, "worlds", world_file_name])
    existing_gazebo_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path = os.pathsep.join(
        path for path in [os.path.join(v550_ackermann_gazebo_dir, "models"), existing_gazebo_model_path] if path
    )
    existing_gazebo_plugin_path = os.environ.get("GAZEBO_PLUGIN_PATH", "")
    gazebo_plugin_path = os.pathsep.join(
        path
        for path in [
            os.path.join(v550_ackermann_gazebo_prefix, "lib", "v550_ackermann_gazebo"),
            existing_gazebo_plugin_path,
        ]
        if path
    )

    lifecycle_nodes = [
        "map_server",
        "amcl",
        "controller_server",
        "planner_server",
        "smoother_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    common_parameters = [params_file, {"use_sim_time": use_sim_time}]
    bt_navigator_parameters = [
        params_file,
        {
            "use_sim_time": use_sim_time,
            "default_nav_to_pose_bt_xml": nav_to_pose_bt_xml,
            "default_nav_through_poses_bt_xml": nav_through_poses_bt_xml,
        },
    ]

    return LaunchDescription(
        [
            SetEnvironmentVariable("GAZEBO_MODEL_PATH", gazebo_model_path),
            SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", gazebo_plugin_path),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "tile_windows",
                default_value="true",
                description="Tile Gazebo left and RViz right across the desktop work area",
            ),
            DeclareLaunchArgument("pause", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "world",
                default_value="warehouse_demo/" + ROBOT_MODEL + ".model",
                description="Industrial warehouse world under v550_ackermann_gazebo/worlds",
            ),
            DeclareLaunchArgument(
                "dynamic_obstacles",
                default_value="true",
                description="Move physically bounded forklift, AGV, and worker obstacles",
            ),
            DeclareLaunchArgument(
                "dynamic_speed_scale",
                default_value="0.8",
                description="Speed multiplier for dynamic industrial obstacles",
            ),
            DeclareLaunchArgument(
                "nav2_startup_timeout",
                default_value="90.0",
                description="Max seconds to wait for Gazebo services, odom/scan, and odom TF before starting Nav2",
            ),
            DeclareLaunchArgument(
                "odom_tf_bridge",
                default_value="true",
                description="Publish odom->base_footprint from /odom with the dedicated TF bridge",
            ),
            DeclareLaunchArgument(
                "urdf_file",
                default_value=default_urdf_file,
                description="Robot URDF used by robot_state_publisher in the industrial navigation scene",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(v550_ackermann_gazebo_dir, "config", "industrial_nav2_params.yaml"),
                description="Nav2 parameters using Smac Hybrid-A* planner",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(v550_ackermann_gazebo_dir, "maps", "industrial_warehouse.yaml"),
                description="Static warehouse occupancy map for Nav2",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(v550_ackermann_gazebo_dir, "rviz", "industrial_nav2.rviz"),
                description="RViz config for industrial Nav2 demo",
            ),
            DeclareLaunchArgument(
                "nav_to_pose_bt_xml",
                default_value=os.path.join(
                    v550_ackermann_gazebo_dir,
                    "behavior_trees",
                    "ackermann_navigate_to_pose_w_replanning_and_recovery.xml",
                ),
                description="Ackermann-compatible NavigateToPose behavior tree",
            ),
            DeclareLaunchArgument(
                "nav_through_poses_bt_xml",
                default_value=os.path.join(
                    v550_ackermann_gazebo_dir,
                    "behavior_trees",
                    "ackermann_navigate_through_poses_w_replanning_and_recovery.xml",
                ),
                description="Ackermann-compatible NavigateThroughPoses behavior tree",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
                launch_arguments={"world": world, "pause": pause}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")),
                condition=IfCondition(gui),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([launch_file_dir, "/robot_state_publisher.launch.py"]),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "urdf_file": urdf_file,
                }.items(),
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="dynamic_obstacles.py",
                name="warehouse_dynamic_obstacles",
                output="screen",
                condition=IfCondition(dynamic_obstacles),
                parameters=[
                    {
                        "enabled": True,
                        "use_sim_time": use_sim_time,
                        "speed_scale": dynamic_speed_scale,
                        "update_rate": 20.0,
                        "robot_clearance": 1.15,
                        "robot_name": ROBOT_MODEL,
                        "obstacle_clearance": 0.45,
                        "prediction_topic": "predicted_dynamic_obstacles",
                        "prediction_frame": "map",
                        "prediction_horizon": 3.0,
                        "prediction_step": 0.25,
                        "prediction_padding": 0.05,
                        "set_entity_state_service": "auto",
                        "model_states_topic": "auto",
                        "service_wait_timeout": nav2_startup_timeout,
                    }
                ],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[params_file, {"use_sim_time": use_sim_time, "yaml_filename": map_yaml}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=common_parameters,
                remappings=[("initialpose", "initialpose_amcl")],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=common_parameters,
                remappings=[("cmd_vel", "cmd_vel_raw")],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=common_parameters,
            ),
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                output="screen",
                parameters=common_parameters,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=common_parameters,
                remappings=[("cmd_vel", "cmd_vel_raw")],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=bt_navigator_parameters,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=common_parameters,
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                parameters=common_parameters,
                remappings=[
                    ("cmd_vel", "cmd_vel_raw"),
                    ("cmd_vel_smoothed", "cmd_vel_nav"),
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="ackermann_cmd_vel_adapter.py",
                name="ackermann_cmd_vel_adapter",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": "cmd_vel_nav",
                        "output_topic": "cmd_vel",
                        "wheel_base": 0.1432,
                        "max_speed": 1.00,
                        "max_reverse_speed": 0.35,
                        "max_steer": 0.34,
                        "max_steer_rate": 2.5,
                        "min_turning_speed": 0.015,
                        "command_timeout": 0.35,
                        "output_rate": 30.0,
                        "plugin_flips_reverse_steering": True,
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="joint_state_sanitizer.py",
                name="joint_state_sanitizer",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": "joint_states_raw",
                        "output_topic": "joint_states",
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="initial_pose_gazebo_sync.py",
                name="initial_pose_gazebo_sync",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": "initialpose",
                        "amcl_topic": "initialpose_amcl",
                        "odom_topic": "odom",
                        "robot_name": ROBOT_MODEL,
                        "set_entity_state_service": "auto",
                        "position_tolerance": 0.05,
                        "yaw_tolerance": 0.08,
                        "sync_timeout": 5.0,
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="trajectory_history.py",
                name="trajectory_history",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "odom_topic": "odom",
                        "reset_topic": "initialpose",
                        "path_topic": "trajectory_history",
                        "sample_distance": 0.03,
                        "sample_yaw": 0.05,
                        "reset_tolerance": 0.10,
                        "max_points": 10000,
                        "z_offset": 0.08,
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="goal_distance_display.py",
                name="goal_distance_display",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "goal_topic": "goal_pose",
                        "odom_topic": "odom",
                        "distance_topic": "goal_distance_cm",
                        "marker_topic": "goal_distance_marker",
                        "plan_topic": "plan",
                        "goal_tolerance": 0.03,
                        "publish_rate": 10.0,
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="odom_tf_broadcaster.py",
                name="odom_tf_broadcaster",
                output="screen",
                condition=IfCondition(odom_tf_bridge),
                parameters=[
                    {
                        "use_sim_time": False,
                        "odom_topic": "odom",
                        "parent_frame": "odom",
                        "child_frame": "base_footprint",
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom_ground_truth",
                output="screen",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "map",
                    "--child-frame-id",
                    "odom",
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": False,
                        "node_names": lifecycle_nodes,
                        "bond_timeout": 4.0,
                    }
                ],
            ),
            Node(
                package="v550_ackermann_gazebo",
                executable="nav2_startup_gate.py",
                name="nav2_startup_gate",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "startup_timeout": nav2_startup_timeout,
                        "gazebo_state_service": "auto",
                        "required_topics": ["odom", "scan"],
                        "target_frame": "odom",
                        "source_frame": "base_footprint",
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
                condition=IfCondition(rviz),
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            TimerAction(
                period=4.0,
                actions=[
                    ExecuteProcess(
                        cmd=[window_tiler, "--timeout", "30"],
                        output="screen",
                        condition=IfCondition(tile_windows),
                    )
                ],
            ),
        ]
    )
