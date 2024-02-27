#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int64, Bool
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import typing as T


class Navigator(BaseNavigator):

    def __init__(self, kpx: float = 2, kpy: float = 2, kdx: float = 2, kdy: float = 2,
                 V_max: float = 1, om_max: float = 2) -> None:
        super().__init__("headingController")
        self.kp = 4
        self.V_PREV_THRES = 0.0001
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def load_traj(self, times: np.ndarray, traj: np.ndarray) -> None:
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        """ Compute only orientation target (used for NavMode.ALIGN and NavMode.Park)

        Returns:
            TurtleBotControl: control target
        """
        err = goal.theta - state.theta
        if err > np.pi:
            err = err - 2 * np.pi
        elif err < -np.pi:
            err = err + 2 * np.pi
            
        om = self.kp * err

        return TurtleBotControl(omega = om)

    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """

        dt = t - self.t_prev
        th = state.theta
        x_d = scipy.interpolate.splev(t, plan.path_x_spline, der=0)
        y_d = scipy.interpolate.splev(t, plan.path_y_spline, der=0)
        xd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=1)
        yd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=1)
        xdd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=2)
        ydd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=2)
        
        ########## Code starts here ##########
        
        if abs(self.V_prev)<V_PREV_THRES:
            self.V_prev = V_PREV_THRES
        v = self.V_prev

        vx = v*np.cos(th)
        vy = v*np.sin(th)
        J = np.array([[np.cos(th), -vy], 
                      [np.sin(th), vx]])
        virtual_control = np.zeros((2,))
        virtual_control[0] = xdd_d + self.kdx*(xd_d-vx) + self.kpx*(x_d-x)
        virtual_control[1] = ydd_d + self.kdy*(yd_d-vy) + self.kpy*(y_d-y)
        controls = np.linalg.solve(J, virtual_control)
        v_new = controls[0]*dt + v

        if abs(v_new)<V_PREV_THRES:
            V = V_PREV_THRES
        else:
            V = v_new
        om = controls[1]

        ########## Code ends here ##########

        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return TurtleBotControl(v = V, omega = om)

    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """

        v_desired = 0.15
        spline_alpha = 0.08

        statespace_lo = (state.x - horizon, state.y - horizon)
        statespace_hi = (state.x + horizon, state.y + horizon)
        astar = AStar(statespace_lo, statespace_hi, (state.x, state.y), (goal.x, goal.y), occupancy, resolution)
        path = astar.reconstruct_path()
        if path is None:
            return None
        ts = [0]*len(path)
        x_vals = [path[0][0]]
        y_vals = [path[0][1]]
        path_x_spline = None
        path_y_spline = None
    
        for i in range(1,len(path)):
            cur_state = np.array(path[i])
            x_vals.append(cur_state[0])
            y_vals.append(cur_state[1])
    
            prev_state = np.array(path[i-1])
            dt = np.sqrt(np.sum((cur_state-prev_state)**2))/v_desired
            ts[i] = ts[i-1] + dt
        
        path_x_spline = scipy.interpolate.splrep(x=ts, y=x_vals, s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(x=ts, y=y_vals, s=spline_alpha)

        return TrajectoryPlan(
            path=pathArray,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
       
        occ = self.occupancy
        return occ.is_free(x)
    
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        
        # move east
        new_state = self.snap_to_grid((x[0] + self.resolution, x[1]))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move west
        new_state = self.snap_to_grid((x[0] - self.resolution, x[1]))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move south
        new_state = self.snap_to_grid((x[0], x[1] - self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)
        
        # move north
        new_state = self.snap_to_grid((x[0], x[1] + self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move NW
        new_state = self.snap_to_grid((x[0] - self.resolution, x[1] + self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move NE
        new_state = self.snap_to_grid((x[0] + self.resolution, x[1] + self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move SW
        new_state = self.snap_to_grid((x[0] - self.resolution, x[1] - self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        # move SE
        new_state = self.snap_to_grid((x[0] + self.resolution, x[1] - self.resolution))
        if self.occupancy.is_free(new_state) and new_state != x:
            neighbors.append(new_state)

        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        
        MAX_VALUE = 1e10

        while len(self.open_set)>0:
            cur_node = self.find_best_est_cost_through()
            if cur_node == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            else:
                self.open_set.remove(cur_node)
                self.closed_set.add(cur_node) # Use closed set to avoid looping paths
                children = self.get_neighbors(cur_node)
                for child in children:
                    if child in self.closed_set:
                        continue
                    elif child not in self.open_set:
                        self.open_set.add(child)
                        self.cost_to_arrive[child] = MAX_VALUE
                
                    if self.cost_to_arrive[child] >= self.cost_to_arrive[cur_node] + self.distance(child, cur_node):
                        self.cost_to_arrive[child] = self.cost_to_arrive[cur_node] + self.distance(child, cur_node)
                        self.came_from[child] = cur_node
                        self.est_cost_through[child] = self.cost_to_arrive[child] + self.distance(child, self.x_goal)
        return False
    
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))


if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Navigator()  # instantiate the node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
