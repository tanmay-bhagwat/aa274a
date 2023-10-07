#!/usr/bin/env python3

import rclpy
import numpy as np
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):

    kp=2.0

    def __init__(self, node_name: str = "heading_controller") -> None:
        super().__init__(node_name)
        
    
    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        om = HeadingController.kp*err
        control = TurtleBotControl()
        control.omega = om
        return control

if __name__=="__main__":
    rclpy.init()
    controller = HeadingController()
    rclpy.spin(controller)
    rclpy.shutdown()