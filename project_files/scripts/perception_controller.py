#!/usr/bin/env python3
from rclpy.node import Node
import rclpy
import numpy as np
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool


class PerceptionController(BaseHeadingController):
    def __init__(self) -> None:
        super().__init__("PerceptionController")  # initialize base class
        self.image_detected = False
        self.kp =2.0
        self.img_sub = self.create_subscription(Bool, "/detector_bool", self.callback, 10)

    # @property
    # def image_detected(self):
    #     return self.get_parameter("image_detected").value
    
    def callback(self, msg: Bool):
        if msg.data:
            self.image_detected = True

    def compute_control_with_goal(self, TurtleBot_current_state:TurtleBotState,TurtleBot_desired_state:TurtleBotState)-> TurtleBotControl: 
        ContrlMess= TurtleBotControl()
        if not self.image_detected:
            ContrlMess.omega = 0.2
        else:
            ContrlMess.omega = 0.0
        
        
        return ContrlMess
    
if __name__ == '__main__':
    rclpy.init()
    HC_Class = PerceptionController()
    rclpy.spin(HC_Class)
    rclpy.shutdown()
