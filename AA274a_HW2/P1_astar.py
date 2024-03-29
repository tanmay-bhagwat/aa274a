import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments

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
        # raise NotImplementedError("is_free not implemented")
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
        # raise NotImplementedError("distance not implemented")
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
        # raise NotImplementedError("get_neighbors not implemented")

        # Easier way than this hard coding? Nested for-loops?
        # for i in range(-1,2):
        #     for j in range(-1,2):
        #         new_state = self.snap_to_grid((x[0] + i*self.resolution, x[1] + j*self.resolution))
        #         if self.is_free(new_state) and new_state != x:
        #             neighbors.append(new_state)
        
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
        # Remember that this is only the geometric path (in C-space) and may not be dynamically feasible (since no control inputs used!)
        # Hence, we would then pass this geometric path from A* to the trajectory opt solver to get control inputs 
        # and even discard/ add new states
        
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
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
        # raise NotImplementedError("solve not implemented")
        
        # Might have been more general to solve using Bellman's criterion and from goal state back to initial state
        # In that case,  only need cost_to_go from query node to goal node to decide optimal path
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
