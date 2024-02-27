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
        u1=xdd_d+self.kpx*(x_d-state.x)+self.kdx*(xd_d-self.V_prev*np.cos(th))
        u2=ydd_d+self.kpy*(y_d-state.y)+self.kdy*(yd_d-self.V_prev*np.sin(th))

        V= self.V_prev+(np.cos(th)*u1+np.sin(th)*u2)*dt
        if(V<self.V_PREV_THRES):
            V=self.V_PREV_THRES
        om=((-u1*np.sin(th)+u2*np.cos(th))/V)
        ########## Code ends here ##########

        """
        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)
        """

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
        if not astar.solve():
            return None
        elif len(astar.path) < 4:
            return None
        self.t_prev = 0
        self.V_prev = 0

        ts = []
        current_t = 0
        for i in range(len(astar.path)):
            if i == 0:
                ts.append(current_t)
            else:
                current_t += np.linalg.norm(np.array(astar.path[i])- np.array(astar.path[i - 1])) / v_desired
                ts.append(current_t)
        
        pathArray = np.asarray(astar.path)
        x_path = pathArray[:, 0]
        y_path = pathArray[:, 1]
        
        path_x_spline = scipy.interpolate.splrep(ts, x_path, k=3, s = spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, y_path, k=3, s = spline_alpha)
        
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
        ########## Code starts here ########
        # With margins
        """
        if self.occupancy.is_free(x) == False:
            # inside an obstacle
            return False
        elif x[0] < self.statespace_lo[0] + self.occupancy.width * .01 or \
             x[1] < self.statespace_lo[1] + self.occupancy.height * .01 or \
             x[0] > self.statespace_hi[0] - self.occupancy.width * .01 or \
             x[1] > self.statespace_hi[1] - self.occupancy.height * .01:
            # at end of grid
            return False
        return True
        """
        # Without margins
        
        if self.occupancy.is_free(np.array(x)) == False:
            # inside an obstacle
            return False
        elif x[0] < self.statespace_lo[0] or \
             x[1] < self.statespace_lo[1] or \
             x[0] > self.statespace_hi[0] or \
             x[1] > self.statespace_hi[1]:
            # at end of grid
            return False
        return True
        
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
        return np.linalg.norm(np.array(x1) -  np.array(x2))
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

        candidates = []

        # Get 8 possible neighbors in all 8 possible directions
        deltas = [-self.resolution, 0, self.resolution]
        for delta1 in deltas:
            for delta2 in deltas:
                candidates.append(self.snap_to_grid((x[0] + delta1, x[1] + delta2)))
        # remove possibility caused by for loops of neighbor being equal to original
        del candidates[4]

        # only add free candidates to neighbors
        for i in range(len(candidates)):
            if self.is_free(np.asarray(candidates[i])):
                neighbors.append(candidates[i])

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

    """
    def plot_path(self, fig_num=0, show_init_label=True):
        #Plots the path found in self.path and the obstacles
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)
    """

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

        # first 4 initialization steps already done in __init__

        while len(self.open_set) > 0:
            # let x_current be the state with the lowest est_cost_through
            current = self.find_best_est_cost_through()
            
            # if x_current is the goal state, reconstruct path
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            
            # remove x_current from open set
            self.open_set.remove(current)

            # add x_current to closed set
            self.closed_set.add(current)

            # iterate over all neighbors of current
            for neighbor in self.get_neighbors(current):

                # if neighbor is in the closed set already, skip it
                if neighbor in self.closed_set:
                    continue

                # tentative cost to arrive at neighbor is cost to arrive to current + distance(current, neighbor)
                tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current, neighbor)

                # if neighbor is not in the open set, add it to the open set; otherwise, if the calculated tentative cost to arrive
                # at the neighbor is more than the existing cost to arrive at the neighbor, skip this iteration; it's a worse path
                if (neighbor in self.open_set) == False:
                    self.open_set.add(neighbor)
                elif neighbor in self.cost_to_arrive.keys():
                    if tentative_cost_to_arrive > self.cost_to_arrive[neighbor]:
                        continue

                # neighbor came from current
                self.came_from[neighbor] = current

                # cost to arrive to neighbor is the tentative cost we calculated
                self.cost_to_arrive[neighbor] = tentative_cost_to_arrive

                # estimated cost to goal through neighbor is tentative cost to arrive to neighbor plus distance from neighbor to goal
                self.est_cost_through[neighbor] = tentative_cost_to_arrive + self.distance(neighbor, self.x_goal)
            

        # if path wasn't reconstructed, something went wrong
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