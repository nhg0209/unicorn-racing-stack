-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- /* Author: Darby Lim */

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 10, -- increase when the lidar is slow and the car is fast
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,  -- increased from the default 0.2 s
  submap_publish_period_sec = 0.3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
  publish_tracked_pose = true, 
  pose_publish_period_sec = 1e-2,
  publish_to_tf = true,
}

MAP_BUILDER.use_trajectory_builder_2d = true -- for 2d slam or localization
MAP_BUILDER.num_background_threads = 6 -- performance tuning via multi-threading

TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 10.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true 
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)
-- TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 5 --0.01
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 0.1 --25
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 10

POSE_GRAPH.constraint_builder.min_score = 0.60
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.80

TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}
POSE_GRAPH.optimize_every_n_nodes = 3 -- should be lower than for mapping

POSE_GRAPH.global_sampling_ratio = 0.003 -- 0.003 -- global matching sampling ratio
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 3. -- linear search window (m)
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(30.) -- angular search window (deg)

-- localization-only configuration
-- POSE_GRAPH.global_constraint_search_after_n_seconds = 10. -- global constraint search period

return options