import numpy as np
from scipy.interpolate import splrep

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
        """

        occ = self.occupancy
        return occ.is_free(x)

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    

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
        """
        neighbors = []
        
        print(x[0], x[1])
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
        """
        
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
    

    def smoothen(self, v_desired=0.15):
        path = self.path
        ts=[0]*len(path)
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
        
        spline_alpha = 0.05
        path_x_spline = splrep(x=ts, y=x_vals, s=spline_alpha)
        path_y_spline = splrep(x=ts, y=y_vals, s=spline_alpha)

        return path_x_spline, path_y_spline, ts[-1]
