""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        max_range = lidar.max_range

        filtered_distances = distances[distances < max_range]
        filtered_angles = angles[distances < max_range]

        # Estimer les positions des d´etections du laser dans le rep`ere absolu
        x_list = np.cos(filtered_angles + pose[2]) * filtered_distances + pose[0]
        y_list = np.sin(filtered_angles + pose[2]) * filtered_distances + pose[1]

        # Convertir ces positions en index de cellules dans la grille d’occupation et supprimer les points hors de la carte,

        x_indices, y_indices = self.grid.conv_world_to_map(x_list, y_list)

        # Supprimer les points hors de la carte
        valid_indices = np.logical_and(np.logical_and(x_indices >= 0, x_indices < self.grid.x_max_map),
                                       np.logical_and(y_indices >= 0, y_indices < self.grid.y_max_map))

        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]

        # Lire et additionner les valeurs des cellules correspondantes dans la carte pour calculer le score.

        score = np.sum(self.grid.occupancy_map[x_indices, y_indices])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # Compose map<-odom and odom<-robot transforms in 2D.
        # If no reference is provided, use the current one stored in the object.
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        odom_pose = np.asarray(odom_pose, dtype=float)
        odom_pose_ref = np.asarray(odom_pose_ref, dtype=float)

        x_ref, y_ref, theta_ref = odom_pose_ref
        x_odom, y_odom, theta_odom = odom_pose

        c = np.cos(theta_ref)
        s = np.sin(theta_ref)

        x_map = x_ref + c * x_odom - s * y_odom
        y_map = y_ref + s * x_odom + c * y_odom
        theta_map = theta_ref + theta_odom

        theta_map = np.arctan2(np.sin(theta_map), np.cos(theta_map))

        corrected_pose = np.array([x_map, y_map, theta_map])

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0
        pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, pose)
        best_pose = self.odom_pose_ref.copy()

        i = 0

        while i < 100:
            # Générer une pose candidate en ajoutant un bruit gaussien à la pose actuelle
            candidate_pose = pose + np.random.normal(0, 10, size=3)

            # Calculer le score de la pose candidate
            score = self._score(lidar, candidate_pose)

            # Si le score de la pose candidate est meilleur que le meilleur score actuel, mettre à jour la meilleure pose et le meilleur score
            if score > best_score:
                best_score = score
                best_pose = candidate_pose

            i += 1
        
        self.odom_pose_ref = best_pose

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        mask = distances < lidar.max_range
        distances = distances[mask]
        angles = angles[mask]

        # Conversion polaire local du laser/ cartésien absolu dans la carte
        x_list = np.cos(angles + pose[2]) * distances + pose[0]
        y_list = np.sin(angles + pose[2]) * distances + pose[1]

        # Update des points sur la ligne avec proba faible
        for x, y in zip(x_list, y_list):
            self.grid.add_value_along_line(pose[0], pose[1], x, y, val=-0.95)

        # Update des points cibles avec proba forte
        self.grid.add_map_points(x_list, y_list, val=4)

        self.grid.add_map_points(x_list + 0.5, y_list + 0.5, val=4)
        self.grid.add_map_points(x_list - 0.5, y_list + 0.5, val=4)

        # Seuil des probas
        np.clip(self.grid.occupancy_map, -20, 20, out=self.grid.occupancy_map)




