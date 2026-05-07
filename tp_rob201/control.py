""" A set of robotics control functions """

import random
import numpy as np


def _wrap_to_pi(angle):
    """Wrap any angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    sensor_angles = lidar.get_ray_angles()

    for i in range(len(laser_dist)):
        if sensor_angles[i] > -np.pi/4 and sensor_angles[i] < np.pi/4 and laser_dist[i] < 50:
            rotation_speed = random.uniform(-1.0, 1.0)
            command = {"forward": 0.0,
                       "rotation": rotation_speed}
            return command

    speed = 0.5
    rotation_speed = 0.0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    current_pose = np.asarray(current_pose, dtype=float)
    goal_pose = np.asarray(goal_pose, dtype=float)

    # Attractice potential field

    K = 10.0  # attractive potential gain

    distance_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])

    # Tolerance to goal
    T = 10
    if distance_to_goal < T:
        command = {"forward": 0.0,
                   "rotation": 0.0}
        return command
    else:
        gradient = K * (goal_pose[:2] - current_pose[:2]) / distance_to_goal

     # Repulsive potential field
    L = 100000.0  # repulsive potential gain
    d_min = 20  # minimum distance to obstacle

    for i in range(len(lidar.get_sensor_values())):
        sensor_dist = lidar.get_sensor_values()[i]
        if sensor_dist < d_min and sensor_dist > 0.01: 
            angle = lidar.get_ray_angles()[i]
            f_rep = L * (1/sensor_dist - 1/d_min) * np.array([np.cos(angle), np.sin(angle)])
            gradient += f_rep
    
    distance_gain = 0.02
    angle_gain = 1.5

    desired_heading = np.arctan2(gradient[1], gradient[0])
    heading_error = _wrap_to_pi(desired_heading - current_pose[2])

    # Slow down linear motion when the goal direction is not in front of the robot.
    forward_speed = distance_gain * distance_to_goal * np.cos(heading_error)
    rotation_speed = angle_gain * heading_error
    
    # Clamp values to valid ranges [-1, 1]
    forward_speed = np.clip(forward_speed, -0.5, 0.5)
    rotation_speed = np.clip(rotation_speed, -0.5, 0.5)
    
    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command
