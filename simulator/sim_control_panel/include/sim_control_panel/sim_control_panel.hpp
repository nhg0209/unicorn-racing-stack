#ifndef SIM_CONTROL_PANEL__SIM_CONTROL_PANEL_HPP_
#define SIM_CONTROL_PANEL__SIM_CONTROL_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>

class QPushButton;

class QLabel;

namespace sim_control_panel
{

// RViz dockable panel for driving the f1tenth_gym_ros sim:
//   - Remove opponent     -> std_msgs/Empty   on /sim/remove_opponent
//   - Opponent speed +/-  -> std_msgs/Float32 on /sim/opp_speed_delta
//   - Drive mode / lidar toggles -> /sim/opp_mode, /sim/{ego,opp}_lidar_enable
// (Spawn an opponent with RViz's "2D Goal Pose" tool -> /goal_pose.)
class SimControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit SimControlPanel(QWidget * parent = nullptr);
  void onInitialize() override;

private Q_SLOTS:
  void onRemoveOpponent();
  void onSpeedUp();
  void onSpeedDown();
  void onModeManual();
  void onModePath();
  void onModeFtg();
  void onToggleEgoLidar();
  void onToggleOppLidar();

private:
  void publishSpeedDelta(float delta);
  void publishMode(const char * mode);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr remove_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ego_lidar_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr opp_lidar_pub_;
  QPushButton * ego_lidar_btn_ {nullptr};
  QPushButton * opp_lidar_btn_ {nullptr};

  QLabel * status_label_ {nullptr};
  double speed_step_ {0.5};
};

}  // namespace sim_control_panel

#endif  // SIM_CONTROL_PANEL__SIM_CONTROL_PANEL_HPP_
