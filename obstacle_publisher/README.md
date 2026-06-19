# Obstacle Publisher

ROS2 package for publishing dynamic and static obstacles in the F1TENTH simulator.

## Nodes

### static_obstacle_manager
- Place obstacles by clicking **"Publish Point"** in RViz
- Clear all obstacles by clicking the green **"Clear Obstacles"** marker at (0, -5)
- Launched automatically with the simulator (`low_level.launch.xml sim:=true`)

### dynamic_obstacle_publisher
- Follows a trajectory CSV (or circular dummy path if no CSV exists)
- Supports reactive lateral oscillation mode

## Launch

**Dynamic obstacle (standalone):**
```bash
ros2 launch obstacle_publisher dynamic_obstacle_publisher.launch.xml map:=<map_name>
```

**With custom parameters:**
```bash
ros2 launch obstacle_publisher dynamic_obstacle_publisher.launch.xml map:=<map_name> \
  speed_scaler:=0.7 reactive:=true reactive_freq:=0.3
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trajectory_csv` | `centerline.csv` | Trajectory CSV file to follow |
| `speed_scaler` | `0.7` | Speed multiplier |
| `constant_speed` | `false` | Use constant speed |
| `reactive` | `false` | Enable lateral oscillation |
| `reactive_freq` | `0.3` | Oscillation frequency (Hz) |
| `starting_s` | `0.0` | Starting position on trajectory (m) |
| `obstacle_length_m` | `0.6` | Obstacle length (m) |
| `obstacle_width_m` | `0.3` | Obstacle width (m) |
| `obstacle_diameter_m` | `0.5` | Static obstacle diameter (m) |
