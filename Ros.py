from rclpy.node import Node

from eufs_msgs.msg import CarState
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64 #For car steering

from eufs_msgs.msg import ConeArrayWithCovariance

from std_msgs.msg import Int16

#from fmdv_msgs.msg import MonitorData, VehicleState, ControlCommand, ConesList, PointsList, Cone
from geometry_msgs.msg import Point

import numpy as np
import pandas as pd

class Ros(Node):

    def __init__(self):
        super().__init__('node') 
        self.x, self.y, self.z = 0,0,0
        self.v_x, self.v_y, self.v_z = 0,0,0        
        self.v = 0 
        self.yaw = 0
        self.currentSteer = 0
        self.xCheck = [0,0,0]       #For monitor -> x, y, yaw
        self.show_monitor = True

        self.conesSeen, self.prevConesSeen = [], []

        self.subsciber_carState = self.create_subscription(CarState, '/ground_truth/state', self.updateCarstate, 10)
        self.subsciber_cones    = self.create_subscription(ConeArrayWithCovariance, "ground_truth/cones", self.updateCones, 10)

        self.subsciber_carSteer = self.create_subscription(Float64, '/ground_truth/car_steer', self.updateCarSteer, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped,'cmd',10)
        # if self.show_monitor:
        #     self.monitor_pub = self.create_publisher(MonitorData, "/fmdv/monitor_data", 10)
        self.publisher_resetsim = self.create_publisher(Int16,'/ros_can/resetsim',10)

    def updateCarstate(self, Message):
        self.x = Message.pose.pose.position.x
        self.y = Message.pose.pose.position.y
        self.z = Message.pose.pose.position.z

        self.v_x = Message.twist.twist.linear.x
        self.v_y = Message.twist.twist.linear.y
        self.v_z = Message.twist.twist.linear.z
        self.v = np.sqrt(self.v_x**2+self.v_y**2+self.v_z**2)

        #Angular velocities - Not used as of now
        # self.angv_x = Message.twist.twist.angular.x
        # self.angv_y = Message.twist.twist.angular.y
        # self.angv_z = Message.twist.twist.angular.z
        # self.angv = np.sqrt(self.angv_x**2+self.angv_y**2+self.angv_z**2)

        #self.slip_angle = Message.slip_angle

        x = Message.pose.pose.orientation.x
        y = Message.pose.pose.orientation.y
        z = Message.pose.pose.orientation.z
        w = Message.pose.pose.orientation.w

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        self.yaw = yaw_z

        self.xCheck = [self.x, self.y, self.yaw]

    def updateCarSteer(self, Message):
        self.currentSteer = Message.data

    def updateCones(self,data):
        """Updates the cone locations in the car frame as seen by the camera"""

        conesSeen = []
        for i in data.blue_cones:
            conesSeen.append([i.point.x,i.point.y,"Blue", "Left"])
        for i in data.yellow_cones:
            conesSeen.append([i.point.x,i.point.y,"Yellow", "Right"])
        for i in data.big_orange_cones:
            if (i.point.y < 0):
                conesSeen.append([i.point.x,i.point.y,"Big Orange", "Right"])
            else:
                conesSeen.append([i.point.x,i.point.y,"Big Orange", "Left"])
        
        if (conesSeen == self.prevConesSeen):
            return
        self.prevConesSeen = self.conesSeen
        self.conesSeen = conesSeen

    def inputingValue(self, acc, steering_angle):
        message = AckermannDriveStamped()
        message.drive.acceleration  = acc  # why not AckermannDriveStamped().drive.acceleration = 1.0
        message.drive.steering_angle = steering_angle
        self.publisher.publish(message)

    # def update_monitor(self, acceleration, steering_angle, localisation, waypoints): #Completely from their code
        
    #     vehicle_state = VehicleState()
    #     vehicle_state.x = float(self.x)
    #     vehicle_state.y = float(self.y)
    #     vehicle_state.yaw = float(self.yaw)
    #     vehicle_state.velocity = float(self.v)
    #     vehicle_state.steer_angle = float(self.currentSteer)

    #     control_command = ControlCommand()
    #     control_command.acceleration = float(acceleration)
    #     control_command.steer_angle = float(steering_angle)

    #     cones_seen = ConesList()

    #     cones = localisation.getNeighbouringCones(self.xCheck)

    #     for cone in cones:
    #         cone_msg = Cone()
    #         cone_msg.x = float(cone[0])
    #         cone_msg.y = float(cone[1])
    #         cone_msg.type = str(cone[2])
    #         cones_seen.cones.append(cone_msg)

    #     map_msg = PointsList()

    #     for sector in localisation.sectorsBlue:
    #         for cone in sector:
    #             point = Point()
    #             point.x = float(cone[0])
    #             point.y = float(cone[1])
    #             map_msg.points.append(point)
        
    #     for sector in localisation.sectorsYellow:
    #         for cone in sector:
    #             point = Point()
    #             point.x = float(cone[0])
    #             point.y = float(cone[1])
    #             map_msg.points.append(point)
            
    #     waypoints_msg = PointsList()

    #     for waypoint in waypoints.waypoints: #idk yet
    #         point = Point()
    #         point.x = float(waypoint[0])
    #         point.y = float(waypoint[1])
    #         waypoints_msg.points.append(point)

    #     monitor_data_msg = MonitorData()
    #     monitor_data_msg.vehicle_state = vehicle_state
    #     monitor_data_msg.control_command = control_command
    #     monitor_data_msg.cones_seen = cones_seen
    #     monitor_data_msg.map = map_msg
    #     monitor_data_msg.waypoints_list = waypoints_msg

    #     self.monitor_pub.publish(monitor_data_msg)

    def resetSim(self, status):
       msg = Int16()
       msg.data = status #0 to reset
       self.publisher_resetsim.publish(msg)
       print('Reset successful')

        #t0 = +2.0 * (w * x + y * z)
        #t1 = +1.0 - 2.0 * (x * x + y * y)
        #roll_x = math.atan2(t0, t1)
    
        #t2 = +2.0 * (w * y - z * x)
        #t2 = +1.0 if t2 > +1.0 else t2
        #t2 = -1.0 if t2 < -1.0 else t2
        #pitch_y = math.asin(t2)
        #return yaw_z #roll_x,pitch_y,yaw_z

        #t3 = +2.0 * (w * z + x * y)
        #t4 = +1.0 - 2.0 * (y * y + z * z)
        #yaw_z = np.arctan2(t3, t4)
        #self.yaw = yaw_z