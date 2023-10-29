#!/usr/bin/env python3

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import rclpy
import numpy as np
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from scipy.interpolate import splev
from astar import AStar


class Navigator(BaseNavigator):

    kp = 2.0
    V_prev = 0.0
    t_prev = 0.0

    def __init__(self, kpx: float = 1., kpy: float = 1., kdx: float = 1., kdy: float = 1.,
                 V_max: float = 0.5, om_max: float = 1):
        super().__init__("navigator_node")
        self.V_PREV_THRES = 0.0001
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
    
    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        om = Navigator.kp*err
        control = TurtleBotControl()
        control.omega = om
        return control
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState,
                                            plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        
        # Assuming that this function is going to be called on consecutive timestamps, so that
        # dt = current t -  prev_t
        dt = t - Navigator.t_prev
        x_d = splev(t, plan.path_x_spline, der=0)
        y_d = splev(t, plan.path_y_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        th_d = float(np.arctan2(yd_d, xd_d))

        if abs(Navigator.V_prev)<self.V_PREV_THRES:
            Navigator.V_prev = self.V_PREV_THRES
        V = Navigator.V_prev

        x = state.x
        y = state.y
        th = state.theta
        vx = V*np.cos(th)
        vy = V*np.sin(th)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        virtual_control = np.zeros((2,))
        virtual_control[0] = xdd_d + self.kdx*(xd_d-vx) + self.kpx*(x_d-x)
        virtual_control[1] = ydd_d + self.kdy*(yd_d-vy) + self.kpy*(y_d-y)

        J = np.array([[np.cos(th), -vy],[np.sin(th), vx]])
        result = np.linalg.solve(J, virtual_control)

        V = result[0]*dt + V
        if abs(V)<self.V_PREV_THRES:
            V = self.V_PREV_THRES
        Navigator.t_prev = t

        desired_state = TurtleBotState()
        th_d = float(np.arctan2(yd_d, xd_d))
        desired_state.theta = th_d
        
        # Do we return omega calculated from inv(J)*virtual_control, or the compute_control output?
        # Doing the second here, but confirm once!
        # om_actual = self.compute_heading_control(state, desired_state)
        om_actual = result[1]
        control = TurtleBotControl()
        control.v = V
        control.omega = om_actual
        
        return control

    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D, resolution: float, 
                                horizon: float) -> TrajectoryPlan | None:
        
        x_init = np.array([state.x, state.y])
        x_goal = np.array([goal.x, goal.y])
        astar = AStar(x_init=x_init, x_goal=x_goal, occupancy=occupancy, 
                      resolution=resolution, statespace_lo=(x_init[0] - horizon, x_init[1] - horizon), 
                      statespace_hi=(x_init[0] + horizon, x_init[1] + horizon))
        
        if not astar.solve() or len(astar.path)<4:
            return 
        
        Navigator.V_prev = 0
        Navigator.t_prev = 0
        path = np.asarray(astar.path)

        path_x_spline, path_y_spline, duration = astar.smoothen()

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=duration
        )
    

if __name__=='__main__':
    rclpy.init()

    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()