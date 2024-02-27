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

    def compute_control_with_goal(self, TurtleBot_current_state:TurtleBotState,TurtleBot_desired_state:TurtleBotState)-> TurtleBotControl: 
        
        heading_error = TurtleBot_desired_state.theta - TurtleBot_current_state.theta
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi
             
        ContrlMess= TurtleBotControl()
        ContrlMess.omega = self.kp *heading_error
        
        return ContrlMess
    
if __name__ == '__main__':
    rclpy.init()
    HC_Class = HeadingController()
    rclpy.spin(HC_Class)
    rclpy.shutdown()
