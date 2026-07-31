#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.clock import Clock, ClockType
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


@dataclass(frozen=True)
class Route:
    name: str
    points: tuple
    z: float
    speed: float
    radius: float
    wait_time: float = 0.0


ROUTES = (
    Route(
        name="dynamic_forklift_1",
        points=((-6.35, -8.65), (6.10, -8.65), (6.10, -8.05), (-6.35, -8.05)),
        z=0.05,
        speed=0.34,
        radius=0.90,
        wait_time=0.8,
    ),
    Route(
        name="dynamic_forklift_2",
        points=((6.10, 8.65), (-6.35, 8.65), (-6.35, 8.05), (6.10, 8.05)),
        z=0.05,
        speed=0.32,
        radius=0.90,
        wait_time=0.8,
    ),
    Route(
        name="dynamic_agv_1",
        points=((-2.65, -2.35), (2.65, -2.35), (2.65, 2.35), (-2.65, 2.35)),
        z=0.10,
        speed=0.36,
        radius=0.55,
        wait_time=0.45,
    ),
    Route(
        name="dynamic_worker_1",
        points=((-7.10, -5.30), (-7.10, 5.30), (-6.70, 5.30), (-6.70, -5.30)),
        z=0.05,
        speed=0.20,
        radius=0.34,
        wait_time=0.8,
    ),
)


STATIC_KEEP_OUTS = (
    (-7.98, -7.72, -10.00, 10.00),
    (7.72, 7.98, -10.00, 10.00),
    (-7.85, 7.85, 9.72, 9.98),
    (-7.85, 7.85, -9.98, -9.72),
    (-6.65, -3.65, 6.12, 6.98),
    (-2.85, 0.15, 6.12, 6.98),
    (2.10, 5.10, 6.12, 6.98),
    (-6.20, -3.20, 3.02, 3.88),
    (1.90, 4.90, 3.02, 3.88),
    (-5.98, -5.12, -1.50, 1.50),
    (5.32, 6.18, -1.50, 1.50),
    (-6.20, -3.20, -3.88, -3.02),
    (1.90, 4.90, -3.88, -3.02),
    (-6.50, -3.50, -6.88, -6.02),
    (-1.90, 1.10, -6.88, -6.02),
    (2.90, 5.90, -6.88, -6.02),
    (-4.60, -3.40, 4.55, 5.65),
    (6.10, 7.00, 4.95, 5.75),
    (-6.20, -5.00, -5.35, -4.35),
    (5.85, 6.85, -5.55, -4.55),
    (-1.55, -0.55, 4.55, 5.35),
    (0.95, 1.75, -5.65, -4.85),
    (-3.80, -2.70, 5.05, 5.65),
    (3.95, 4.95, 1.45, 2.45),
    (-2.70, -2.00, -5.55, -4.85),
    (0.60, 1.60, 4.70, 5.40),
)


@dataclass
class RouteState:
    route: Route
    segment_index: int
    x: float
    y: float
    yaw: float
    wait_remaining: float = 0.0


def advance_prediction_state(state, duration, speed):
    """Advance a copied route state without robot or obstacle interactions."""
    remaining = max(float(duration), 0.0)
    speed = max(float(speed), 0.0)
    while remaining > 1.0e-6:
        if state.wait_remaining > 0.0:
            waited = min(state.wait_remaining, remaining)
            state.wait_remaining -= waited
            remaining -= waited
            continue

        route = state.route
        start = route.points[state.segment_index]
        end = route.points[(state.segment_index + 1) % len(route.points)]
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        distance = math.hypot(end[0] - state.x, end[1] - state.y)
        if distance <= 1.0e-6:
            state.x, state.y = end
            state.yaw = yaw
            state.segment_index = (state.segment_index + 1) % len(route.points)
            state.wait_remaining = route.wait_time
            continue

        if speed <= 1.0e-6:
            break
        travel = min(speed * remaining, distance)
        state.x += math.cos(yaw) * travel
        state.y += math.sin(yaw) * travel
        state.yaw = yaw
        remaining -= travel / speed
        if travel >= distance - 1.0e-6:
            state.x, state.y = end
            state.segment_index = (state.segment_index + 1) % len(route.points)
            state.wait_remaining = route.wait_time

    return state.x, state.y


def build_prediction_points(
    route_states,
    moving,
    speed_scale,
    horizon,
    step,
    padding,
    point_spacing,
):
    points = []
    for name, state in route_states.items():
        centers = [(state.x, state.y)]
        if moving.get(name, False) and horizon > 0.0:
            predicted = RouteState(
                route=state.route,
                segment_index=state.segment_index,
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                wait_remaining=state.wait_remaining,
            )
            speed = state.route.speed * max(speed_scale, 0.0)
            elapsed = step
            while elapsed <= horizon + 1.0e-6:
                centers.append(advance_prediction_state(predicted, step, speed))
                elapsed += step

        radius = state.route.radius + padding
        samples = max(8, math.ceil(2.0 * math.pi * radius / point_spacing))
        for x, y in centers:
            points.append((x, y, 0.20))
            for index in range(samples):
                angle = 2.0 * math.pi * index / samples
                points.append(
                    (
                        x + radius * math.cos(angle),
                        y + radius * math.sin(angle),
                        0.20,
                    )
                )

    return points


def circle_intersects_box(x, y, radius, box):
    min_x, max_x, min_y, max_y = box
    closest_x = min(max(x, min_x), max_x)
    closest_y = min(max(y, min_y), max_y)
    return math.hypot(x - closest_x, y - closest_y) <= radius


class DynamicObstacles(Node):
    def __init__(self):
        super().__init__("warehouse_dynamic_obstacles")
        self.declare_parameter("enabled", True)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("update_rate", 15.0)
        self.declare_parameter("model_prefix", "")
        self.declare_parameter("robot_name", "wheeltec_v550_ackermann")
        self.declare_parameter("robot_clearance", 0.95)
        self.declare_parameter("obstacle_clearance", 0.25)
        self.declare_parameter("set_entity_state_service", "auto")
        self.declare_parameter("model_states_topic", "auto")
        self.declare_parameter("service_wait_timeout", 60.0)
        self.declare_parameter("diagnostic_period", 2.5)
        self.declare_parameter("prediction_topic", "predicted_dynamic_obstacles")
        self.declare_parameter("prediction_frame", "map")
        self.declare_parameter("prediction_horizon", 3.0)
        self.declare_parameter("prediction_step", 0.25)
        self.declare_parameter("prediction_padding", 0.05)
        self.declare_parameter("prediction_point_spacing", 0.22)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.model_prefix = str(self.get_parameter("model_prefix").value)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.robot_clearance = max(float(self.get_parameter("robot_clearance").value), 0.0)
        self.obstacle_clearance = max(float(self.get_parameter("obstacle_clearance").value), 0.0)
        self.requested_state_service = str(self.get_parameter("set_entity_state_service").value)
        self.requested_model_states_topic = str(self.get_parameter("model_states_topic").value)
        self.service_wait_timeout = max(float(self.get_parameter("service_wait_timeout").value), 1.0)
        self.diagnostic_period = max(float(self.get_parameter("diagnostic_period").value), 0.5)
        prediction_topic = str(self.get_parameter("prediction_topic").value)
        self.prediction_frame = str(self.get_parameter("prediction_frame").value)
        self.prediction_horizon = max(
            float(self.get_parameter("prediction_horizon").value), 0.0
        )
        self.prediction_step = max(
            float(self.get_parameter("prediction_step").value), 0.05
        )
        self.prediction_padding = max(
            float(self.get_parameter("prediction_padding").value), 0.0
        )
        self.prediction_point_spacing = max(
            float(self.get_parameter("prediction_point_spacing").value), 0.05
        )
        update_rate = max(float(self.get_parameter("update_rate").value), 1.0)
        self.period = 1.0 / update_rate
        self.last_update = None
        self.robot_xy = None
        self.client = None
        self.state_service_name = None
        self.model_state_sub = None
        self.connected = False
        self.disabled_after_timeout = False
        self.start_time = time.monotonic()
        self.last_diagnostic_time = 0.0

        self.route_states = {
            route.name: RouteState(
                route=route,
                segment_index=0,
                x=route.points[0][0],
                y=route.points[0][1],
                yaw=math.atan2(
                    route.points[1][1] - route.points[0][1],
                    route.points[1][0] - route.points[0][0],
                ),
            )
            for route in ROUTES
        }

        self.prediction_publisher = self.create_publisher(
            PointCloud2, prediction_topic, 10
        )
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(self.period, self.on_timer, clock=self.wall_clock)
        self.get_logger().info(
            "Dynamic warehouse obstacles "
            f"enabled={self.enabled} speed_scale={self.speed_scale:.2f}; "
            f"prediction_horizon={self.prediction_horizon:.1f}s; "
            "waiting for Gazebo state interfaces"
        )

    def on_model_states(self, msg):
        try:
            index = msg.name.index(self.robot_name)
        except ValueError:
            return
        pose = msg.pose[index]
        self.robot_xy = (float(pose.position.x), float(pose.position.y))

    def on_timer(self):
        if not self.enabled:
            return

        if not self.connected:
            self.try_connect_gazebo()
            return

        now = self.get_clock().now()
        if self.last_update is None:
            self.last_update = now
            return
        dt = min(max((now - self.last_update).nanoseconds / 1e9, 0.0), 0.2)
        self.last_update = now

        moving = {}
        for state in self.route_states.values():
            x, y, yaw, vx, vy = self.advance_route_state(state, dt)
            self.set_model_state(state.route.name, x, y, state.route.z, yaw, vx, vy)
            moving[state.route.name] = math.hypot(vx, vy) > 1.0e-3
        self.publish_predictions(moving)

    def try_connect_gazebo(self):
        service_name = self.resolve_service_name()
        topic_name = self.resolve_model_states_topic()
        missing = []

        if service_name is None:
            missing.append("gazebo_msgs/srv/SetEntityState service")
        elif self.client is None:
            self.client = self.create_client(SetEntityState, service_name)
            self.state_service_name = service_name
            self.get_logger().info(f"Using Gazebo state service {service_name}")

        if topic_name is None:
            missing.append("gazebo_msgs/msg/ModelStates topic")
        elif self.model_state_sub is None:
            self.model_state_sub = self.create_subscription(
                ModelStates,
                topic_name,
                self.on_model_states,
                10,
            )
            self.get_logger().info(f"Using Gazebo model states topic {topic_name}")

        if self.client is not None and not self.client.wait_for_service(timeout_sec=0.0):
            missing.append(self.state_service_name or "Gazebo SetEntityState service")

        if missing:
            self.log_waiting_status(missing)
            if time.monotonic() - self.start_time > self.service_wait_timeout:
                self.enabled = False
                self.disabled_after_timeout = True
                self.get_logger().error(
                    "Timed out waiting for Gazebo state interfaces; "
                    "dynamic obstacles are disabled for this run. "
                    f"Missing: {', '.join(missing)}. "
                    f"Visible services/topics: {self.visible_gazebo_interfaces()}"
                )
            return

        self.connected = True
        self.last_update = None
        for state in self.route_states.values():
            self.set_model_state(
                state.route.name,
                state.x,
                state.y,
                state.route.z,
                state.yaw,
                0.0,
                0.0,
            )
        self.publish_predictions({name: False for name in self.route_states})
        self.get_logger().info("Gazebo state interfaces ready; dynamic obstacle motion started")

    def publish_predictions(self, moving):
        points = build_prediction_points(
            self.route_states,
            moving,
            self.speed_scale,
            self.prediction_horizon,
            self.prediction_step,
            self.prediction_padding,
            self.prediction_point_spacing,
        )

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.prediction_frame
        self.prediction_publisher.publish(
            point_cloud2.create_cloud_xyz32(header, points)
        )

    def resolve_service_name(self):
        requested = self.requested_state_service.strip()
        if requested and requested.lower() != "auto":
            return requested

        services = self.get_service_names_and_types()
        return self.select_name_by_type(
            services,
            "gazebo_msgs/srv/SetEntityState",
            ("/gazebo/set_entity_state", "/set_entity_state"),
        )

    def resolve_model_states_topic(self):
        requested = self.requested_model_states_topic.strip()
        if requested and requested.lower() != "auto":
            return requested

        topics = self.get_topic_names_and_types()
        return self.select_name_by_type(
            topics,
            "gazebo_msgs/msg/ModelStates",
            ("/gazebo/model_states", "/model_states"),
        )

    @staticmethod
    def select_name_by_type(names_and_types, type_name, preferred_names):
        matches = sorted(
            name
            for name, types in names_and_types
            if type_name in types
        )
        if not matches:
            return None
        for preferred_name in preferred_names:
            if preferred_name in matches:
                return preferred_name
        return matches[0]

    def log_waiting_status(self, missing):
        now = time.monotonic()
        if now - self.last_diagnostic_time < self.diagnostic_period:
            return
        self.last_diagnostic_time = now
        self.get_logger().info(
            "Waiting for Gazebo state interfaces: "
            f"{', '.join(missing)}. "
            f"Visible services/topics: {self.visible_gazebo_interfaces()}"
        )

    def visible_gazebo_interfaces(self):
        service_names = sorted(
            name
            for name, _ in self.get_service_names_and_types()
            if "gazebo" in name or "entity_state" in name
        )
        topic_names = sorted(
            name
            for name, _ in self.get_topic_names_and_types()
            if "gazebo" in name or "model_states" in name
        )
        return f"services={service_names or 'none'}, topics={topic_names or 'none'}"

    def advance_route_state(self, state, dt):
        route = state.route
        speed = max(route.speed * max(self.speed_scale, 0.0), 0.0)
        if speed <= 1e-3:
            return state.x, state.y, state.yaw, 0.0, 0.0

        if state.wait_remaining > 0.0:
            state.wait_remaining = max(0.0, state.wait_remaining - dt)
            return state.x, state.y, state.yaw, 0.0, 0.0

        start = route.points[state.segment_index]
        end = route.points[(state.segment_index + 1) % len(route.points)]
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        distance_remaining = math.hypot(end[0] - state.x, end[1] - state.y)
        step = min(speed * dt, distance_remaining)

        if distance_remaining <= 1e-4:
            state.x = end[0]
            state.y = end[1]
            state.yaw = yaw
            state.segment_index = (state.segment_index + 1) % len(route.points)
            state.wait_remaining = route.wait_time
            return state.x, state.y, state.yaw, 0.0, 0.0

        next_x = state.x + math.cos(yaw) * step
        next_y = state.y + math.sin(yaw) * step
        if self.is_blocked(route.name, next_x, next_y, route.radius):
            return state.x, state.y, yaw, 0.0, 0.0

        state.x = next_x
        state.y = next_y
        state.yaw = yaw
        if step >= distance_remaining - 1e-4:
            state.x = end[0]
            state.y = end[1]
            state.segment_index = (state.segment_index + 1) % len(route.points)
            state.wait_remaining = route.wait_time
            return state.x, state.y, state.yaw, 0.0, 0.0

        return state.x, state.y, state.yaw, speed * math.cos(yaw), speed * math.sin(yaw)

    def is_blocked(self, name, x, y, radius):
        if any(circle_intersects_box(x, y, radius, box) for box in STATIC_KEEP_OUTS):
            return True

        for other_name, other in self.route_states.items():
            if other_name == name:
                continue
            min_distance = radius + other.route.radius + self.obstacle_clearance
            if math.hypot(x - other.x, y - other.y) < min_distance:
                return True

        if self.robot_xy is not None:
            min_robot_distance = radius + self.robot_clearance
            if math.hypot(x - self.robot_xy[0], y - self.robot_xy[1]) < min_robot_distance:
                return True

        return False

    def set_model_state(self, name, x, y, z, yaw, vx, vy):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw * 0.5)
        pose.orientation.w = math.cos(yaw * 0.5)

        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)

        request = SetEntityState.Request()
        state = request.state
        state.name = self.model_prefix + name
        state.pose = pose
        state.twist = twist
        state.reference_frame = "world"
        self.client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacles()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
