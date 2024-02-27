#!/usr/bin/env python3

import numpy as np
import typing as T
from scipy.signal import convolve2d

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from nav_msgs.msg import OccupancyGrid

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D


"""
Trigger explore via command line publish to /nav_success
OK to explore on /nav_success True or False
    Actually might need to consider picking a different frontier state on replan
In explore, if current state is None, just pick a random frontier state
"""


class FrontierExplorer(Node):
    def __init__(self):
        # self.get_logger().debug("Entering FrontierExplorer __init__")
        super().__init__("frontier_explore_node") # name node

        self.occupancy:  T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        """
        self.state = TurtleBotState()
        self.state.x = 0.0
        self.state.y = 0.0
        self.state.theta = 0.0
        """

        # subscribe and publish to interface with navigator
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.on_nav_success_pub, 10)
        self.goal_state_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)


        # subscribe to /state and /map topics
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.on_state_pub, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.on_map_pub, 10)

        # self.get_logger().debug("Finishing FrontierExplorer __init__")

        self.initial_goal()

        '''
        myBool = Bool()
        myBool.data = True
        self.on_nav_success_pub(myBool)
        '''

    def initial_goal(self) -> None:
        new_turtle_goal = TurtleBotState()
        new_turtle_goal.x = 100.0
        new_turtle_goal.y = 100.0
        new_turtle_goal.theta = 3.14
        self.goal_state_pub.publish(new_turtle_goal)

        """
        turtle_goal2 = TurtleBotState()
        turtle_goal2.x = 100.0
        turtle_goal2.y = 100.0
        turtle_goal2.theta = 3.14
        self.goal_state_pub.publish(turtle_goal2)
        """

    # Callback on subscriber to /nav_success published from navigator
    def on_nav_success_pub(self, nav_success_status: Bool) -> None:
        # self.get_logger().debug("Entering on_nav_success_pub")
        if nav_success_status.data or not nav_success_status.data: # currently doing it regardless. change back
            # robot reached commanded navigation pose
            # give a new state goal
            new_state_goal = self.explore()
            new_turtle_goal = TurtleBotState()
            new_turtle_goal.x = new_state_goal[0]
            new_turtle_goal.y = new_state_goal[1]
            new_turtle_goal.theta = 0.0 # TODO: consider how to set angle
            self.goal_state_pub.publish(new_turtle_goal)

        else:
            # planning or re-planning failed
            # TODO
            pass

    # callback on subscriber to /state
    def on_state_pub(self, state: TurtleBotState) -> None:
        # self.get_logger().debug("Entering on_state_pub")
        self.state = state

    # callback on subscriber to /map
    def on_map_pub(self, msg: OccupancyGrid) -> None:
        # self.get_logger().debug("Entering on_map_pub")
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy = np.array([msg.info.width, msg.info.height]),
            origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size = 9,
            probs=msg.data
        )

    def explore(self):
        """ Returns next state to explore
        """


        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################


        # 1) Create binary masks for unknown, occupied, and free cells
        unknown_cells_mask = self.occupancy.probs == -1
        occupied_cells_mask = np.zeros((self.occupancy.size_xy[1], self.occupancy.size_xy[0]))
        free_cells_mask = np.zeros((self.occupancy.size_xy[1], self.occupancy.size_xy[0]))

        # inefficient but functional for loop over all cells
        for cell_y in range(0, self.occupancy.size_xy[1]):
            for cell_x in range(0, self.occupancy.size_xy[0]):
                if self.occupancy.probs[cell_y, cell_x] == -1:
                    continue
                if self.occupancy.probs[cell_y, cell_x] < self.occupancy.thresh:
                    free_cells_mask[cell_y, cell_x] = 1
                else:
                    occupied_cells_mask[cell_y, cell_x] = 1

        # 2) Create counts of unknown, occ, unocc in surrounding window for each cell
        kernel = np.ones((window_size, window_size))

        unknown_counts = convolve2d(unknown_cells_mask, kernel, mode='same')
        occupied_counts = convolve2d(occupied_cells_mask, kernel, mode='same')
        free_counts = convolve2d(free_cells_mask, kernel, mode='same')

        # 3) Create percentages for each cell

        # todo: clean up edge cases?
        kernel_area = window_size ** 2
        unknown_percentages = unknown_counts / kernel_area
        free_percentages = free_counts / kernel_area

        # 4) Create good masks for unk, occ, and unocc
        good_unknown_cells = unknown_percentages >= 0.2
        good_occupied_cells = occupied_counts == 0
        good_free_cells = free_percentages >= 0.3

        # 5) AND them
        good_cells = np.logical_and(np.logical_and(good_unknown_cells, good_occupied_cells), good_free_cells)

        # 6) Get good grid cells and convert to states
        frontier_states = None
        for cell_y in range(0, self.occupancy.size_xy[1]):
            for cell_x in range(0, self.occupancy.size_xy[0]):
                if good_cells[cell_y, cell_x]:
                    cur_state = self.occupancy.grid2state(np.array([cell_x, cell_y]))
                    if frontier_states is None:
                        frontier_states = cur_state
                    else:
                        frontier_states = np.vstack((frontier_states, cur_state))

        # part (ii): find the closest state
        cur_state = np.array([self.state.x, self.state.y])
        distances = np.linalg.norm(cur_state - frontier_states, axis=1)
        min_index = np.argmin(distances)
        min_dist_state = frontier_states[min_index, :]
        return min_dist_state


if __name__ == "__main__":
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()
