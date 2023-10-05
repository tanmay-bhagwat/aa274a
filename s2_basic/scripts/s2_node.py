#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class S2Node(Node):

    def __init__(self):
        super().__init__("s2_node")
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odomsub_callback, 10)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        timer_period = 1
        # self.timer=self.create_timer(timer_period, self.timer_callback)
        
    def odomsub_callback(self, odom_obj):
        xval, yval = odom_obj.pose.pose.position.x, odom_obj.pose.pose.position.y
        self.get_logger().info(f"Turtlebot (X,Y) = ({xval}, {yval})")
        # print(f"Turtlebot (X,Y) = ({xval}, {yval})")

    def timer_callback(self):
        vel_obj = Twist()
        vel_obj.linear.x = 0.2
        self.vel_pub.publish(vel_obj)
        self.get_logger().info(f"Publishing cmd_vel {vel_obj.linear.x}")
        # print(f"Publishing cmd_vel {vel_obj.linear.x}")


def main(args=None):
    rclpy.init(args=None)
    s2_node = S2Node()
    rclpy.spin(s2_node)


if __name__=="__main__":
    main()
