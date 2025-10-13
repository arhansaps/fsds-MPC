from scipy.optimize import minimize

import pandas as pd
import numpy as np
import rclpy
import time
from rclpy.node import Node

from eufs_msgs.msg import CarState
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int16
#from eufs_msgs.msg import Reset_Status

#run only this file, every file ive copy pasted required functions

Waypoints = pd.read_csv("/home/abhishek/FM/src/Stanley/waypointposn.csv")

class mpc(Node):
    def __init__(self):
        self.currentWaypoint = 0
        self.prev_flag = -1
        self.v_max = 10
        self.lap_start = time.time()
        self.v_sum = 0
        self.v_count = 0
        self.previous_error = 0

        self.lap_time = 26

        super().__init__('node') 
        self.x, self.y, self.z = 0,0,0
        self.v_x, self.v_y, self.v_z = 0,0,0        
        self.v = [0,0,0]        
        self.yaw = 0

        self.subsciber_carstate = self.create_subscription(CarState, '/ground_truth/state', self.listener_callback_carstate, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped,'cmd',10)
        self.publisher_resetsim = self.create_publisher(Int16,'/ros_can/resetsim',10)

        self.constfn = 0

    def calc_avg_vel(self, velocity):
        self.v_sum += velocity
        self.v_count += 1
    
    def ifCrossedCone(self, car_x, car_y, update = True):
        X = list(Waypoints.iloc[:,0])
        Y = list(Waypoints.iloc[:,1])
        #self.currentWaypoint = self.currentWaypoint % 38 #As there are only 37 way points (starting from 0)
        delX = X[(self.currentWaypoint+1)%38] - X[self.currentWaypoint%38]
        delY = Y[(self.currentWaypoint+1)%38] - Y[self.currentWaypoint%38]

        flag = delX*car_x + delY*car_y - delX*X[self.currentWaypoint%38] - delY*Y[self.currentWaypoint%38] # equation of straight line perpendicular to line of way points

        if self.prev_flag*flag < 0 and update is True:
            if (self.currentWaypoint+1)%38 == 0:
                self.lap_end = time.time()
                print("Lap time - ",self.lap_end - self.lap_start)
                self.lap_time = self.lap_end - self.lap_start
                print("Average lap velocity = ", self.v_sum/self.v_count)
                self.v_sum = 0
                self.v_count = 0
                self.lap_start = time.time()
            self.currentWaypoint+=1
        if update is False:
            return True
        
    def get_ref_vel(self, car_pos=[0,0], vel=None):
        if vel is None:
            vel = self.v_max

        tetha1 = np.arctan((car_pos[1] - Waypoints.iloc[(self.currentWaypoint%38)][1])/(car_pos[0] - Waypoints.iloc[(self.currentWaypoint%38)][0]))
        tetha2 = np.arctan((car_pos[1] - Waypoints.iloc[(self.currentWaypoint+1)%38,:][1])/(car_pos[0] - Waypoints.iloc[(self.currentWaypoint+1)%38,:][0]))
        tetha3 = np.arctan((car_pos[1] - Waypoints.iloc[(self.currentWaypoint+2)%38,:][1])/(car_pos[0] - Waypoints.iloc[(self.currentWaypoint+2)%38,:][0]))

        tetha = 0.7*tetha1 + 0.2*tetha2 +0.1*tetha3
        v_ref = np.abs(vel*np.cos(tetha)**(1/5))
        return v_ref

    

    def listener_callback_carstate(self, Message):
        self.x = Message.pose.pose.position.x
        self.y = Message.pose.pose.position.y
        self.z = Message.pose.pose.position.z

        self.v_x = Message.twist.twist.linear.x
        self.v_y = Message.twist.twist.linear.y
        self.v_z = Message.twist.twist.linear.z
        self.v = [self.v_x, self.v_y, self.v_z]

        x = Message.pose.pose.orientation.x
        y = Message.pose.pose.orientation.y
        z = Message.pose.pose.orientation.z
        w = Message.pose.pose.orientation.w

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        self.yaw = yaw_z

    def inputingValue(self, acc, steering_angle):
        message = AckermannDriveStamped()
        message.drive.acceleration  = acc  # why not AckermannDriveStamped().drive.acceleration = 1.0
        message.drive.steering_angle = steering_angle
        self.publisher.publish(message)

    def normalize(self, delta):
        if delta > np.pi:
            delta = -1*delta + np.pi
        if delta < -1*np.pi:
            delta = -1*delta + np.pi        
        return delta

    def get_predicted_errors(self, x, y, yaw):
        X = list(Waypoints.iloc[:,0])
        Y = list(Waypoints.iloc[:,1])
        currentWaypoint = self.currentWaypoint

        if self.ifCrossedCone(x, y, False):     #checking if x and y are predicted and not updating counter
            currentWaypoint = self.currentWaypoint + 1
        try:
            delX = X[currentWaypoint%38] - X[(currentWaypoint-1)%38]
            delY = Y[currentWaypoint%38] - Y[(currentWaypoint-1)%38]
        except:
            delX = X[currentWaypoint%38] - 0
            delY = Y[currentWaypoint%38] - 0

        crosstrack_error = (delX*(Y[(currentWaypoint-1)%38]-y) - delY*(X[(currentWaypoint-1)%38]-x))/(delX**2 + delY**2)**0.5
        global_waypoint=np.array([X[currentWaypoint%38], Y[currentWaypoint%38]])
        car_posn=np.array([x, y])
        relative_posn=-car_posn+global_waypoint
        rotation_matrix=np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)]]
            )
        waypoint_posn_in_car=rotation_matrix @ relative_posn
        
        waypoint_heading = np.arctan2(waypoint_posn_in_car[1], waypoint_posn_in_car[0])
        heading_error = waypoint_heading 
        
        '''waypoint_heading = np.arctan2(delY,delX)
        heading_error =  waypoint_heading -  yaw'''
        return crosstrack_error, heading_error
    
    def objective_function(self, params):
        t = 0.1
        a, delta = params

        a_x = a * np.cos(self.yaw)
        a_y = a* np.sin(self.yaw)
        x = self.x + self.v_x*t +0.5*a_x*t**2
        y = self.y + self.v_y*t +0.5*a_y*t**2
        yaw = self.yaw + delta

        crosstrack_error, heading_error = self.get_predicted_errors(x, y, yaw)
        
        current_velocity = (self.v_x**2 + self.v_y**2 + self.v_z**2)

        v = current_velocity + a*t

        v_ref = self.get_ref_vel([x,y])
        error = v_ref - v

        predicted_avg_vel = (self.v_sum + v)/(self.v_count+1)

        #dist_travelled = 

        costfn = np.abs(5.5*crosstrack_error)**2 + np.abs(2*heading_error)**2 + np.abs(3*error)**2 + 0.1/(predicted_avg_vel) #+ 1/dist_travelled
        
        self.previous_error = error
        return costfn
    
    def constraint(self, params):
        a, delta  = params
        return -1*np.pi < delta < np.pi
    
    def calculate(self):
        # Initial guess
        initial_guess = [1,self.yaw]
        #constraints - Idk how to use so haven't used. I put them in the objective function instead.
        constraints = {'type': 'ineq', 'fun': self.constraint}
        # Call the minimize function
        bounds = [(-3, 3), (-np.pi, np.pi)]
        result = minimize(self.objective_function, initial_guess, bounds=bounds)#, method='BFGS')#, constraints=[constraints])
        # Extract the optimal values
        optimal_values = result.x
        throttle = optimal_values[0]
        steering_angle = self.normalize(optimal_values[1])
        return throttle, steering_angle
    

def main(args=None):
    
    rclpy.init(args=args)
    controller = mpc()
    time_period = 1/30 
    start = time.time()

    while True:
        dt = time.time() - start
        if dt >= time_period:
            
            rclpy.spin_once(controller)

            controller.ifCrossedCone(controller.x,controller.y)
            
            velocity = (controller.v[0]**2+controller.v[1]**2+controller.v[2]**2)**0.5
            
            controller.calc_avg_vel(velocity)

            acc, steering_angle = controller.calculate()
            print(steering_angle)
            controller.inputingValue(acc,steering_angle)

            start = time.time()

if __name__ == '__main__':
    main()
