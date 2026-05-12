"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, wall_following_control
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.goal = [-180, 10, 0]
        self.temp_goal = self.goal

        self.final_goal = [0, 0, 0]

        self.grid = self.occupancy_grid

        self.last_distance_to_goal = np.linalg.norm(self.goal[:2] - self.corrected_pose[:2])
        self.last_progresses = np.zeros(50)

        self.escape_protocol_counter = 0

        self.path = None

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1
        return self.control_tp2_extended()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command
    
    def control_tp1_extended(self):
        """
        Control function for TP1
        Control funtion with Wall following behavior
        """

        # Compute new command speed to perform obstacle avoidance
        command = wall_following_control(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            distances = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()
            free_spaces = [(200.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
            if free_spaces:
                chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                      chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                      0])
            else:
                self.goal = self.goal + [-50, 0, 0]  # Move the goal to the left if no free space is found
        
        self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal)

        return command
    
    def control_tp2_extended(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        Implementing a local minima escape strategy by checking if the robot is stuck
        """
        pose = self.odometer_values()

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            distances = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()
            free_spaces = [(200.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
            if free_spaces:
                chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                      chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                      0])

        distance_to_goal = np.linalg.norm(self.goal[:2] - pose[:2])
        
        if self.escape_protocol_counter > 0:
            if self.escape_protocol_counter == 1:
                self.goal = self.temp_goal
            self.escape_protocol_counter -= 1

        elif np.all(self.last_progresses < 0.5) and self.counter % 100 == 0:
            self.temp_goal = self.goal
            distances = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()
            free_spaces = [(200.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
            if free_spaces:
                chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                      chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                      0])
            self.escape_protocol_counter = 150  # Number of steps to execute the escape protocol
        
        elif self.last_distance_to_goal is not None:
            self.last_progresses = np.roll(self.last_progresses, -1)
            self.last_progresses[-1] = self.last_distance_to_goal - distance_to_goal

        self.last_distance_to_goal = distance_to_goal
        
        self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal)

        return command
    
    def control_tp3(self):
        """
        Control function for TP3
        Main control function with full SLAM, random exploration and path planning
        """

        pose = self.odometer_values()

        # Compute new command speed to perform obstacle avoidance
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            distances = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()
            free_spaces = [(200.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
            if free_spaces:
                chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                      chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                      0])
        
        distance_to_goal = np.linalg.norm(self.goal[:2] - pose[:2])
        
        self.last_progresses = np.roll(self.last_progresses, -1)
        self.last_progresses[-1] = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal
        
        if self.counter % 50 == 0:
            if np.all(self.last_progresses < 0.5):
                distances = self.lidar().get_sensor_values()
                angles = self.lidar().get_ray_angles()
                free_spaces = [(180.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
                if free_spaces:
                    chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                    self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                        chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                        0])
        if self.counter % 10 == 0:
            # Update map with new observation
            self.tiny_slam.update_map(self.lidar(), pose)
        
            self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal)

        return command
    
    def control_tp4(self):
        """
        Control function for TP4
        Main control function with full SLAM, random exploration and path planning
        """

        pose = self.odometer_values()

        if self.counter > 10:
            # Localise the robot and update the odometry reference
            score = self.tiny_slam.localise(self.lidar(), pose)

        # Compute new command speed to perform obstacle avoidance
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            distances = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()
            free_spaces = [(180.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0 and (angle > -np.pi/4 and angle < np.pi/4)]
            if free_spaces:
                chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                      chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                      0])
        
        distance_to_goal = np.linalg.norm(self.goal[:2] - pose[:2])
        
        self.last_progresses = np.roll(self.last_progresses, -1)
        self.last_progresses[-1] = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal

        if self.counter % 20 == 0:
            if np.all(np.abs(self.last_progresses) < 0.2):
                distances = self.lidar().get_sensor_values()
                angles = self.lidar().get_ray_angles()
                free_spaces = [(180.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
                if free_spaces:
                    chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                    self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                        chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                        0])

            # Update map with new observation
            self.tiny_slam.update_map(self.lidar(), pose)
        
            self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal)

        return command
    
    def control_tp5(self):
        """
        Control function for TP5
        Main control function with full SLAM, random exploration and path planning
        """

        pose = self.odometer_values()

        if self.counter > 10:
            # Localise the robot and update the odometry reference
            score = self.tiny_slam.localise(self.lidar(), pose)

        if self.counter % 5000 == 0:
            self.path = self.planner.plan(pose, [0, 0, 0])
            if self.path is not None and self.path.shape[1] > 0:
                self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])

        # Compute new command speed to perform obstacle avoidance
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            if self.path is not None and self.path.shape[1] > 0:
                self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])
                self.path = self.path[:, 1:]
            else:
                distances = self.lidar().get_sensor_values()
                angles = self.lidar().get_ray_angles()
                free_spaces = [(180.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0 and (angle > -np.pi/4 and angle < np.pi/4)]
                if free_spaces:
                    chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                    self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                        chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                        0])
        
        distance_to_goal = np.linalg.norm(self.goal[:2] - pose[:2])
        
        self.last_progresses = np.roll(self.last_progresses, -1)
        self.last_progresses[-1] = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal

        if self.counter % 20 == 0:
            if np.all(np.abs(self.last_progresses) < 0.2):
                distances = self.lidar().get_sensor_values()
                angles = self.lidar().get_ray_angles()
                free_spaces = [(180.0, angle) for dist, angle in zip(distances, angles) if dist > 200.0]
                if free_spaces:
                    chosen_space = free_spaces[np.random.choice(len(free_spaces))]
                    self.goal = np.array([chosen_space[0] * np.cos(chosen_space[1] + pose[2]) + pose[0],
                                        chosen_space[0] * np.sin(chosen_space[1] + pose[2]) + pose[1],
                                        0])

            # Update map with new observation
            self.tiny_slam.update_map(self.lidar(), pose)

            self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal, traj=self.path)

        return command
    
    def control_tp6(self):
        """
        Control function for TP6
        Main control function with full SLAM, frontier-based exploration and path planning
        """

        pose = self.odometer_values()

        if self.counter > 10:
            self.tiny_slam.localise(self.lidar(), pose)
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            if self.path is not None and self.path.shape[1] > 0:
                self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])
                self.path = self.path[:, 1:]
            else:
                frontier = self.planner.explore_frontiers()
                if frontier is not None:
                    self.final_goal = frontier
                    self.path = self.planner.plan(pose, [frontier[0], frontier[1], 0])
                    if self.path is not None and self.path.shape[1] > 0:
                        self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])
                else:
                    self.final_goal = [0, 0, 0]
                    self.path = self.planner.plan(pose, self.final_goal)
                    if self.path is not None and self.path.shape[1] > 0:
                        self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])
        
        distance_to_goal = np.linalg.norm(self.goal[:2] - pose[:2])
        
        self.last_progresses = np.roll(self.last_progresses, -1)
        self.last_progresses[-1] = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal

        if self.counter % 20 == 0:
            if np.all(np.abs(self.last_progresses) < 0.2):
                self.path = self.planner.plan(pose, self.final_goal)
                self.goal = np.array([self.path[0, 0], self.path[1, 0], 0]) if self.path is not None and self.path.shape[1] > 0 else self.goal

            # Update map with new observation
            self.tiny_slam.update_map(self.lidar(), pose)

            self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal, traj=self.path)

        return command
