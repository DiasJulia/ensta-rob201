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

        self.goal = [-180, 0, 0]

        self.grid = self.occupancy_grid

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1
        return self.control_tp1_extended()

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
        goal = [-180,0,0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command
    
    def control_tp3(self):
        """
        Control function for TP3
        Main control function with full SLAM, random exploration and path planning
        """

        pose = self.odometer_values()

        # Update map with new observation
        self.tiny_slam.update_map(self.lidar(), pose)

        # Compute new command speed to perform obstacle avoidance
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        if command == {'forward': 0, 'rotation': 0}:
            self.goal = np.random.uniform(low=[-500, -500, 0], high=[500, 140, 0])
        
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

        # Update map with new observation
        self.tiny_slam.update_map(self.lidar(), pose)

        # Compute new command speed to perform obstacle avoidance
    
        command = potential_field_control(self.lidar(), pose, self.goal)

        # potential_goals = self.lidar().get_sensor_values()
        # potential_goals_angles = self.lidar().get_ray_angles()
        # potential_goals = potential_goals[potential_goals < ]
        # potential_goals_angles = potential_goals_angles[potential_goals < 10]

        if command == {'forward': 0, 'rotation': 0}:
            # goal_ref = np.random.choice(len(potential_goals))
            # self.goal = np.array([potential_goals[goal_ref] * np.cos(potential_goals_angles[goal_ref]) + pose[0],
            #                   potential_goals[goal_ref] * np.sin(potential_goals_angles[goal_ref]) + pose[1],
            #                   0])
            self.goal = np.random.uniform(low=[-500, -500, 0], high=[500, 140, 0])
        
        self.occupancy_grid.display_cv(robot_pose=pose, goal=self.goal)

        return command
