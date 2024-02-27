#!/usr/bin/env python3
from rclpy.node import Node
import rclpy
import numpy as np
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class HeadingController(BaseHeadingController):
    def __init__(self) -> None:
        super().__init__("HeadingController")  # initialize base class
        self.kp =2.0

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        om = HeadingController.kp*err
        control = TurtleBotControl()
        control.omega = om
        return control
    
if __name__ == '__main__':
    rclpy.init()
    HC_Class = HeadingController()
    rclpy.spin(HC_Class)
    rclpy.shutdown()
