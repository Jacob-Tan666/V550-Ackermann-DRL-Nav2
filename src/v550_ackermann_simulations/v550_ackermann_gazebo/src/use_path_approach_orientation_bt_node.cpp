#include <algorithm>
#include <cmath>
#include <string>

#include "behaviortree_cpp_v3/action_node.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

namespace v550_ackermann_gazebo
{

class UsePathApproachOrientation : public BT::SyncActionNode
{
public:
  UsePathApproachOrientation(
    const std::string & name,
    const BT::NodeConfiguration & config)
  : BT::SyncActionNode(name, config)
  {
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>(
        "goal", "Original goal whose exact position is preserved"),
      BT::InputPort<nav_msgs::msg::Path>(
        "path", "Collision-free 2D guide path used to determine approach heading"),
      BT::InputPort<double>(
        "heading_baseline", 0.35,
        "Distance before the path end used to calculate the approach heading"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>(
        "adjusted_goal", "Goal with an Ackermann-reachable approach orientation")
    };
  }

private:
  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped goal;
    nav_msgs::msg::Path path;
    double heading_baseline = 0.35;

    if (!getInput("goal", goal)) {
      throw BT::RuntimeError("UsePathApproachOrientation missing input [goal]");
    }
    if (!getInput("path", path)) {
      throw BT::RuntimeError("UsePathApproachOrientation missing input [path]");
    }
    getInput("heading_baseline", heading_baseline);

    // A one-point path means the robot is already at the requested position.
    // Preserve the original orientation; the position-only goal checker will stop it.
    if (path.poses.size() < 2) {
      setOutput("adjusted_goal", goal);
      return BT::NodeStatus::SUCCESS;
    }

    const auto & end = path.poses.back().pose.position;
    const double baseline = std::max(0.05, heading_baseline);
    std::size_t approach_index = path.poses.size() - 2;

    while (approach_index > 0) {
      const auto & point = path.poses[approach_index].pose.position;
      if (std::hypot(end.x - point.x, end.y - point.y) >= baseline) {
        break;
      }
      --approach_index;
    }

    const auto & approach = path.poses[approach_index].pose.position;
    const double dx = end.x - approach.x;
    const double dy = end.y - approach.y;

    if (std::hypot(dx, dy) > 1.0e-6) {
      const double yaw = std::atan2(dy, dx);
      goal.pose.orientation.x = 0.0;
      goal.pose.orientation.y = 0.0;
      goal.pose.orientation.z = std::sin(0.5 * yaw);
      goal.pose.orientation.w = std::cos(0.5 * yaw);
    }

    setOutput("adjusted_goal", goal);
    return BT::NodeStatus::SUCCESS;
  }
};

class UseExactGoalPosition : public BT::SyncActionNode
{
public:
  UseExactGoalPosition(
    const std::string & name,
    const BT::NodeConfiguration & config)
  : BT::SyncActionNode(name, config)
  {
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>(
        "goal", "Goal containing the exact requested position"),
      BT::InputPort<nav_msgs::msg::Path>(
        "input_path", "Kinematically feasible Hybrid-A* path"),
      BT::OutputPort<nav_msgs::msg::Path>(
        "output_path", "Path whose final point is the exact requested position")
    };
  }

private:
  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped goal;
    nav_msgs::msg::Path path;

    if (!getInput("goal", goal)) {
      throw BT::RuntimeError("UseExactGoalPosition missing input [goal]");
    }
    if (!getInput("input_path", path)) {
      throw BT::RuntimeError("UseExactGoalPosition missing input [input_path]");
    }
    if (path.poses.empty()) {
      throw BT::RuntimeError("UseExactGoalPosition received an empty path");
    }

    // Smac paths end at a costmap cell center. Replacing only the last position
    // removes up to one cell of systematic error while retaining its feasible yaw.
    path.poses.back().pose.position = goal.pose.position;
    setOutput("output_path", path);
    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace v550_ackermann_gazebo

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<v550_ackermann_gazebo::UsePathApproachOrientation>(
    "UsePathApproachOrientation");
  factory.registerNodeType<v550_ackermann_gazebo::UseExactGoalPosition>(
    "UseExactGoalPosition");
}
