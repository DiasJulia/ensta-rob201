"""
Planner class
Implementation of A*
"""

import copy
import heapq
import math
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None

    def get_neighbors(self, current_cell):
        """ Return list of free (i.e. not obstacle) neighbour cells 
            with the format of current_cell: (i, j) in the map frame
        """
        neighbor_list = []
        # TODO for TP5: iterate through neighbors and add free ones to neighbor_list
        i, j = current_cell
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid.x_max_map and 0 <= nj < self.grid.y_max_map:
                    neighbor_list.append((ni, nj))

        return neighbor_list

    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """ Return heuristic goal distance """
        h = 0
        # TODO for TP5: compute heuristic distance between cell_1 and cell_2

        h = np.sqrt((cell_1[0] - cell_2[0]) ** 2 + (cell_1[1] - cell_2[1]) ** 2)

        return h

    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)

        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])

        # creates a copy of occupancy map to modify it and take into account
        # a margin in the walls
        self.map_walls = copy.deepcopy(self.grid.occupancy_map)
        # TODO for TP5: dilate walls in self.map_walls to take into account a margin around obstacles

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = (self.map_walls > 0).astype(np.uint8)
        self.map_walls = cv2.dilate(wall_mask, kernel, iterations=1)

        cv2.imshow("map_walls", self.map_walls * 255)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # dictionary to trace back route
        came_from = {}

        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0

        # best guess of cost for each cell (cost + heuristic)
        mu = 3.0  # heuristic weight
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal) * mu

        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                if self.map_walls[cell[0], cell[1]] > 0:
                     continue
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal) * mu
                    heapq.heappush(open_set, (f_score[cell], cell))

        # goal was never reached
        print('failed getting to objective')
        return None

    def explore_frontiers(self):
        """ Frontier based exploration """

        free_threshold = -0.5
        unknown_min_threshold = -0.5
        unknown_max_threshold = 0.5
        
        occupancy_map = self.grid.occupancy_map

        unknown_mask = (occupancy_map > unknown_min_threshold) & (occupancy_map < unknown_max_threshold)
        free_mask = (occupancy_map < free_threshold)

        map_walls = copy.deepcopy(self.grid.occupancy_map)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        wall_mask = (map_walls > 0).astype(np.uint8)
        map_walls = cv2.dilate(wall_mask, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # A frontier is a free cell that has at least one unknown neighbor
        unknown_mask_dilated = cv2.dilate(unknown_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        frontier_mask = free_mask & unknown_mask_dilated & ~map_walls.astype(bool)

        #show frontiers 
        frontier_display = np.zeros_like(occupancy_map)
        frontier_display[frontier_mask] = 1
        cv2.imshow("frontier", frontier_display)

        num_labels, labels = cv2.connectedComponents(frontier_mask.astype(np.uint8))
    
        frontiers = []
        for lbl in range(1, num_labels):
            idx = np.where(labels == lbl)
            if idx[0].size < 20:  # aumentado: filtra pequeno ruído
                continue
            x_map_indices = idx[0].astype(np.int32)
            y_map_indices = idx[1].astype(np.int32)
            centroid_x_map = int(np.mean(x_map_indices))
            centroid_y_map = int(np.mean(y_map_indices))
            wx, wy = self.grid.conv_map_to_world(np.array([centroid_x_map]), np.array([centroid_y_map]))
            frontiers.append((float(wx[0]), float(wy[0])))

        if len(frontiers) == 0:
            return None
        
        # Escolhe o frontier mais próximo (ou maior, conforme preferir)
        robot_pose = self.grid.conv_world_to_map(0, 0)  # pose local do robô no mapa
        distances_to_frontier = [np.sqrt((f[0] - robot_pose[0])**2 + (f[1] - robot_pose[1])**2) for f in frontiers]
        closest_frontier_idx = np.argmin(distances_to_frontier)
        
        goal = frontiers[closest_frontier_idx]
        return goal