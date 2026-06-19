#include "sim_control_panel/sim_control_panel.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFrame>

#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace sim_control_panel
{

SimControlPanel::SimControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout;

  auto * hint = new QLabel(
    "<b>Opponent / Obstacles</b><br/>"
    "Spawn opponent: use the <i>2D Goal Pose</i> tool.");
  hint->setWordWrap(true);
  layout->addWidget(hint);

  auto * obs_row = new QHBoxLayout;
  auto * remove_btn = new QPushButton("Remove opponent");
  auto * clear_obs_btn = new QPushButton("Clear static obstacles");
  obs_row->addWidget(remove_btn);
  obs_row->addWidget(clear_obs_btn);
  layout->addLayout(obs_row);

  auto * line0 = new QFrame;
  line0->setFrameShape(QFrame::HLine);
  line0->setFrameShadow(QFrame::Sunken);
  layout->addWidget(line0);

  layout->addWidget(new QLabel("Opponent drive mode:"));
  auto * mode_row = new QHBoxLayout;
  auto * manual_btn = new QPushButton("Manual");
  auto * path_btn = new QPushButton("Path");
  auto * ftg_btn = new QPushButton("FTG");
  mode_row->addWidget(manual_btn);
  mode_row->addWidget(path_btn);
  mode_row->addWidget(ftg_btn);
  layout->addLayout(mode_row);

  auto * line1 = new QFrame;
  line1->setFrameShape(QFrame::HLine);
  line1->setFrameShadow(QFrame::Sunken);
  layout->addWidget(line1);

  layout->addWidget(new QLabel("LiDAR (off = faster sim):"));
  auto * lidar_row = new QHBoxLayout;
  ego_lidar_btn_ = new QPushButton("Ego LiDAR: ON");
  opp_lidar_btn_ = new QPushButton("Opp LiDAR: ON");
  ego_lidar_btn_->setCheckable(true);
  opp_lidar_btn_->setCheckable(true);
  ego_lidar_btn_->setChecked(true);
  opp_lidar_btn_->setChecked(true);
  lidar_row->addWidget(ego_lidar_btn_);
  lidar_row->addWidget(opp_lidar_btn_);
  layout->addLayout(lidar_row);

  auto * line = new QFrame;
  line->setFrameShape(QFrame::HLine);
  line->setFrameShadow(QFrame::Sunken);
  layout->addWidget(line);

  auto * speed_row = new QHBoxLayout;
  speed_row->addWidget(new QLabel("Opp speed:"));
  auto * down_btn = new QPushButton("-");
  auto * up_btn = new QPushButton("+");
  down_btn->setMaximumWidth(40);
  up_btn->setMaximumWidth(40);
  speed_row->addWidget(down_btn);
  speed_row->addWidget(up_btn);
  layout->addLayout(speed_row);

  status_label_ = new QLabel("");
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  layout->addStretch();
  setLayout(layout);

  connect(remove_btn, &QPushButton::clicked, this, &SimControlPanel::onRemoveOpponent);
  connect(clear_obs_btn, &QPushButton::clicked, this, &SimControlPanel::onClearObstacles);
  connect(up_btn, &QPushButton::clicked, this, &SimControlPanel::onSpeedUp);
  connect(down_btn, &QPushButton::clicked, this, &SimControlPanel::onSpeedDown);
  connect(manual_btn, &QPushButton::clicked, this, &SimControlPanel::onModeManual);
  connect(path_btn, &QPushButton::clicked, this, &SimControlPanel::onModePath);
  connect(ftg_btn, &QPushButton::clicked, this, &SimControlPanel::onModeFtg);
  connect(ego_lidar_btn_, &QPushButton::clicked, this, &SimControlPanel::onToggleEgoLidar);
  connect(opp_lidar_btn_, &QPushButton::clicked, this, &SimControlPanel::onToggleOppLidar);
}

void SimControlPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  remove_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/sim/remove_opponent", 10);
  clear_obstacles_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/sim/clear_obstacles", 10);
  speed_pub_ = node_->create_publisher<std_msgs::msg::Float32>("/sim/opp_speed_delta", 10);
  mode_pub_ = node_->create_publisher<std_msgs::msg::String>("/sim/opp_mode", 10);
  ego_lidar_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/sim/ego_lidar_enable", 10);
  opp_lidar_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/sim/opp_lidar_enable", 10);
}

void SimControlPanel::onRemoveOpponent()
{
  if (remove_pub_) {
    remove_pub_->publish(std_msgs::msg::Empty());
    if (status_label_) {status_label_->setText("Removed opponent");}
  }
}

void SimControlPanel::publishSpeedDelta(float delta)
{
  if (!speed_pub_) {return;}
  std_msgs::msg::Float32 msg;
  msg.data = delta;
  speed_pub_->publish(msg);
  if (status_label_) {
    status_label_->setText(QString("Opp speed %1%2 m/s")
      .arg(delta >= 0 ? "+" : "").arg(static_cast<double>(delta)));
  }
}

void SimControlPanel::onSpeedUp() {publishSpeedDelta(static_cast<float>(speed_step_));}
void SimControlPanel::onSpeedDown() {publishSpeedDelta(static_cast<float>(-speed_step_));}

void SimControlPanel::publishMode(const char * mode)
{
  if (!mode_pub_) {return;}
  std_msgs::msg::String msg;
  msg.data = mode;
  mode_pub_->publish(msg);
  if (status_label_) {status_label_->setText(QString("Opp mode: %1").arg(mode));}
}

void SimControlPanel::onModeManual() {publishMode("manual");}
void SimControlPanel::onModePath() {publishMode("path");}

void SimControlPanel::onModeFtg()
{
  publishMode("ftg");
  // FTG drives off the opponent's lidar -> force it ON (UI + topic).
  if (opp_lidar_btn_) {
    opp_lidar_btn_->setChecked(true);
    opp_lidar_btn_->setText("Opp LiDAR: ON");
  }
  if (opp_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = true;
    opp_lidar_pub_->publish(m);
  }
}

void SimControlPanel::onToggleEgoLidar()
{
  bool on = ego_lidar_btn_->isChecked();
  ego_lidar_btn_->setText(on ? "Ego LiDAR: ON" : "Ego LiDAR: OFF");
  if (ego_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = on;
    ego_lidar_pub_->publish(m);
  }
}

void SimControlPanel::onToggleOppLidar()
{
  bool on = opp_lidar_btn_->isChecked();
  opp_lidar_btn_->setText(on ? "Opp LiDAR: ON" : "Opp LiDAR: OFF");
  if (opp_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = on;
    opp_lidar_pub_->publish(m);
  }
}

}  // namespace sim_control_panel

PLUGINLIB_EXPORT_CLASS(sim_control_panel::SimControlPanel, rviz_common::Panel)
