""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    speed = 0.0
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
    print("Current pose:", int(current_pose[0]), int(current_pose[1]))
    print("Goal pose:", int(goal_pose[0]), int(goal_pose[1]))

    # Attractice potential field

    K = 2.0  # attractive potential gain

    distance_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])

    # Tolerance to goal
    T = 10
    if distance_to_goal < T:
        gradient = np.array([0.0, 0.0])
    else:
        gradient = K * (goal_pose[:2] - current_pose[:2]) / distance_to_goal

     # Repulsive potential field
    L = 1.0  # repulsive potential gain
    d_min = 0.5  # minimum distance to obstacle

    for i in range(len(lidar.get_sensor_values())):
        sensor_dist = lidar.get_sensor_values()[i]
        if sensor_dist < d_min and sensor_dist > 0.01: 
            angle = lidar.get_ray_angles()[i]
            f_rep = L * (1/sensor_dist - 1/d_min) * np.array([np.cos(angle), np.sin(angle)])
            gradient += f_rep
    
    forward_speed = np.linalg.norm(gradient) * 0.5
    rotation_speed = np.arctan2(gradient[1], gradient[0]) * 0.5
    
    # Clamp values to valid ranges [-1, 1]
    forward_speed = np.clip(forward_speed, -1.0, 1.0)
    rotation_speed = np.clip(rotation_speed, -1.0, 1.0)
    
    print(f"Gradient: {gradient}, Forward speed: {forward_speed}, Rotation speed: {rotation_speed}")

    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command
