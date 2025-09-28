import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped as control
from nav_msgs.msg import Odometry as pos

from scipy.optimize import minimize
import csv
import time
import numpy as np
import os
import sys
import time

# Add fsds package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

#/home/ajith/basicController/small_track_shifted.csv
# ~/basicController/small_track_shifted.csv

path_to_wp_Csv= "/home/gitaansh/FM/FSDS_fm/src/MPC/wpsNew.csv" #waypointposn.csv"
class car():
    def __init__(self):
        self.waypoints = list()
        with open(path_to_wp_Csv,"r") as f:
            r = csv.reader(f)
            for i in r:
                self.waypoints.append([float(i[0]),float(i[1])])
        
        #path data
        self.m = self.m_perp = self.theta =  self.prev_op_sign = self.wp_index = 0
        self.theta_MPC = self.m_perp_MPC = self.prev_op_MPC = self.yaw_MPC = 0  #MPC prediction data
        
        self.client = fsds.FSDSClient("172.18.224.1")

        # Check network connection, exit if not connected
        self.client.confirmConnection()

        # After enabling setting trajectory setpoints via the api. 
        self.client.enableApiControl(True)
        #lap timer data
        self.lap_time_start =  None  
        self.total_laps     =  0 

        #car data 
        self.state      = [1,0] 
        self.x          = self.y  =  self.yaw = 0
        self.vel        = [0,0]
        self.WheelBase  = 1.580 #m
        self.cs = None

        # #ROS2 objects
        # self.publisher = self.create_publisher(control, '/cmd', 10)
        # self.car       = control()
        # self.pos       = self.create_subscription(pos,'ground_truth/odom',self.state_update,10)
        # self.publisher.publish(self.car)

    def state_update(self): #callback function for updating the state of the car
       
        q = (self.client.getImuData(imu_name='Imu',vehicle_name='FSCar')).orientation#quarternion data

        t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        t4 = +1.0 - 2.0 * (q.y_val*q.y_val + q.z_val* q.z_val)
        self.yaw = np.arctan2(t3, t4)

        CS= self.client.getCarState(vehicle_name='FSCar')
        CS= CS.kinematics_estimated
        self.cs = CS
        self.vel = [float(CS.linear_velocity.x_val),float(CS.linear_velocity.y_val)]
        self.theta = self.straight_line(self.wp_index)
        self.x = CS.position.x_val
        self.y = CS.position.y_val
        
        
        print(self.x,self.y)
        self.update_waypoint(self.x,self.y)
        print(self.wp_index)
        self.optimize_control()
        return None
        
    def global_to_car_frame(self,x,y,yaw,wp_index): #corrected
        rotation_matrix = np.array([
                                    [np.cos(-yaw), -np.sin(-yaw)],
                                    [np.sin(-yaw), np.cos(-yaw)]
                                ])
    
        points   =  np.array([self.waypoints[wp_index%len(self.waypoints)][0]-x,self.waypoints[wp_index%len(self.waypoints)][1]- y])
        rotated  =  rotation_matrix @ points
        return rotated[0], rotated[1]
    
    def wrap_to_pi(self,yaw):
        # while yaw>np.pi:
        #     yaw -= 2*np.pi
        # while yaw<np.pi:
        #     yaw+=2*np.pi
        # return yaw
        return (yaw + np.pi) % (2 * np.pi) - np.pi
    
    def heading_error(self,x,y,yaw,wp_index): #corrected
        x,y   = self.global_to_car_frame(x,y,yaw,wp_index)
        return abs(np.arctan2(y,x)) 
    
    def distance_to_waypoint(self,x1,y1,wp_index): #corrected
        xy = self.waypoints[(wp_index+1 )%len(self.waypoints)]
        return ((xy[0]-x1)**2 + (xy[1]-y1)**2)**0.5

    def delta_steer(self,steer):
        return abs(steer-self.state[1])
         
    def update_waypoint(self,x,y):              #function for lap timer and waypoint index updation
        x1 = self.waypoints[self.wp_index%len(self.waypoints)][0]
        y1 = self.waypoints[self.wp_index%len(self.waypoints)][1] #checked once
        c  =  y1-((self.m_perp)*x1)
        op = self.m_perp*x + c - y
        print("hi")
        if (self.prev_op_sign>0 and op<0) or (self.prev_op_sign<0 and op>0) or op==0 :
                self.wp_index+=1
                self.straight_line(self.wp_index)
                if self.wp_index%len(self.waypoints) == 1:
                    print("Lap started")
                    if self.lap_time_start is None:
                        self.lap_time_start = time.time()
                    else:
                        lap_time = time.time() - self.lap_time_start
                        print(f"Lap {self.total_laps} completed! Time: {lap_time:.2f} seconds")
                        self.lap_time_start = time.time()
                
                #self.lap_timer()
                print("Current waypoint = ",self.wp_index)           
        self.prev_op_sign = op
        return None
    
    def vel_error(self,theta,v,wp_index):
        if wp_index == 0 or v < 0:
            ref_x_dot  =  2.5#arbitrary value
        else:
            #ref_x_dot = 5.0 * (np.tanh(2 * self.t_avg))
            ref_x_dot = 8 * np.cos(theta)
            #ref_x_dot = max(3, ref_x_dot)
        return abs(ref_x_dot-v)

    def cross_track_error(self,x,y,wp_index,k=1):#corrected
        x0  =   self.waypoints[(wp_index-k)%len(self.waypoints)][0]
        y0  =   self.waypoints[(wp_index-k)%len(self.waypoints)][1]       
        x1  =   self.waypoints[(wp_index-0)%len(self.waypoints)][0]
        y1  =   self.waypoints[(wp_index-0)%len(self.waypoints)][1]
        x2  =   self.waypoints[(wp_index+k)%len(self.waypoints)][0]
        y2  =   self.waypoints[(wp_index+k)%len(self.waypoints)][1]
        A   =   -(y2-y1) #Ax + By = C (straight line std form)
        B   =   (x2-x1)   
        C  =   -(y1-y0)  
        D =     (x1-x0)
        c0 = abs((D*(y0-y) - C*(x0-x))/(C**2 + D**2)**0.5)
        c1 = abs((B*(y1-y) - A*(x1-x))/(A**2 + B**2)**0.5)
        c = 0.3*c0 + 0.7*c1

        return c
    

    def straight_line(self,wp_index,x=1,k=1):
        x2  =  self.waypoints[(wp_index+k)%len(self.waypoints)][0]
        y2  =  self.waypoints[(wp_index+k)%len(self.waypoints)][1]

        x1  =  self.waypoints[wp_index%len(self.waypoints)][0]
        y1  =  self.waypoints[wp_index%len(self.waypoints)][1]
        m   =  (y2-y1)/(x2-x1)

        if x==1:
            self.m       =  m
            self.m_perp  =  -1/self.m
            return
        elif x==2:
            self.m_perp_MPC =  -1/m
            self.theta_MPC  =  np.arctan(m)

            return 
        else:
            return np.arctan(m)
    
    def update_waypoint_predict(self,x,y,wp_index):
        self.straight_line(wp_index,2)
        x1 = self.waypoints[wp_index%len(self.waypoints)][0]
        y1 = self.waypoints[wp_index%len(self.waypoints)][1] #checked once
        c  = y1-((self.m_perp_MPC)*x1)
        op = self.m_perp_MPC*x + c - y
        if (self.prev_op_MPC>0 and op<0) or (self.prev_op_MPC<0 and op>0) or op==0 :
                wp_index+=1                     
        self.prev_op_MPC = op
        #print("predicted = ",wp_index)
        return wp_index
    
    def wrap_steer(self,steer):
        t =self.yaw_MPC 
        # if t>np.pi:
        #     t-=np.pi 
        # if t<-np.pi:    
        #     t+=np.pi
        t = (t+ np.pi) % (2 * np.pi) - np.pi
        return t
    def predict(self, x, y, a, v, steer, t):
        L = self.WheelBase 
        
        # Update velocity with acceleration
        v_mag = np.hypot(v[0], v[1])  # Magnitude of velocity a2+b2=c2
        v_mag += a * t
        v_mag = max(0, v_mag)  

        # Update yaw with steering angle
        beta = np.arctan(0.46 * np.tan(steer))                 # Slip angle approximation Lr/L = 0.46 slip angle formula
        yaw_rate = (v_mag / L) * np.cos(beta) * np.tan(steer)  # Update yaw rate
        self.yaw_MPC += yaw_rate * t
        self.yaw_MPC  = self.wrap_to_pi(self.yaw_MPC)
        #steer -=  yaw_rate * t

        # Update position with updated velocity and yaw
        x += v_mag * np.cos(self.yaw_MPC + beta) * t
        y += v_mag * np.sin(self.yaw_MPC + beta) * t

        # Update velocity vector components
        v[0] = v_mag * np.cos(self.yaw_MPC)
        v[1] = v_mag * np.sin(self.yaw_MPC)
        
        return x, y, v
    
    def dynamic_slip(self,v,steer):
        return 1.0 * (np.arctan(0.46 * np.tan(steer)) - np.arctan2(v[1],v[0]))**2

    def cost_function(self, steer, x, y, wp_index, v):
        #       0      1       2        3       4       5
        w    = [10,    7,      2.8,       10,      3,      1]
        Vx = np.hypot(v[0],v[1])
        # Calculate each error term
        cross_track_error = self.cross_track_error(x, y, wp_index)
        distance_to_waypoint = self.distance_to_waypoint(x, y, wp_index)
        delta_steer = self.delta_steer(steer)
        heading_error = self.heading_error(x, y, self.yaw_MPC, wp_index) 
        vel_error = self.vel_error(self.theta_MPC, Vx, wp_index)
        dynamic_slip = self.dynamic_slip(v,steer)
        # # Print each error term
        # print(f"Waypoint Index: {wp_index}")
        # print(f"Cross Track Error: {cross_track_error}")
        # print(f"Distance to Waypoint: {distance_to_waypoint}")
        # print(f"Delta Steer: {delta_steer}")
        # print(f"Heading Error: {heading_error}")
        # print(f"Velocity Error: {vel_error}")
        
        # Calculate the total cost
        cost = (    
            w[0] * cross_track_error +
            w[1] * distance_to_waypoint +
            w[2] * delta_steer +
            w[3] * heading_error +
            w[4] * vel_error+
            w[5] * dynamic_slip
        )
        
        # Print the total cost
        #print("="*100)
        return cost 

    
    def optimize_function(self,state):
        a     =  state[0]
        steer =  state[1]
        v     =  self.vel
        self.yaw_MPC = self.yaw 

        #prediction 1 
        # s =  np.clip(steer,-0.2,0.2)
        # self.yaw_MPC += s

        x,y,v1      =  self.predict(self.x,self.y,a,v,steer,0.5)
        wp_index   =  self.update_waypoint_predict(x,y,self.wp_index)
        cost1      =  self.cost_function(steer,x,y,wp_index,v)

        #prediction 2
        # s =  np.clip(steer,-0.2,0.2)
        # self.yaw_MPC += s
        x,y,v1    =  self.predict(self.x,self.y,a,v,steer,0.2)
        wp_index  =  self.update_waypoint_predict(x,y,wp_index)
        cost2     =  self.cost_function(steer,x,y,wp_index,v)
        
        #prediction 3
        # s =  np.clip(steer,-0.2,0.2)
        # self.yaw_MPC += s
        # x,y,v     =  self.predict(x,y,a,v,steer,0.1)
        # wp_index  =  self.update_waypoint_predict(x,y,wp_index)
        # cost3     =  self.cost_function(steer,x,y,wp_index,v)
        #final cost
        cost      =  cost1 * 0.6  + cost2 * 0.4 # + cost3 * 0.1
        return cost 
    def acceleration_to_throttle_brake(self,acc, a_max=5.0, b_max=8.0):
        """
        Convert acceleration to throttle and brake (both in [0,1])
        - a_max: max forward acceleration
        - b_max: max deceleration (braking)
        """
        acc = np.clip(acc, -b_max, a_max)

        if acc >= 0:
            throttle = acc / a_max
            brake = 0.0
        else:
            throttle = 0.0
            brake = -acc / b_max
            brake = 1
        return np.clip(throttle, 0.0, 0.2), np.clip(brake, 0.0, 1.0)

    def optimize_control(self):
        initial_guess  =  np.array([self.state[0],self.state[1]] )#Initial guess for scipy
        bounds = ([(-1, 1),(-0.52, 0.52)])
        result         =  minimize(self.optimize_function, initial_guess, method='slsqp',bounds=bounds) #optimizing the cost function
        self.state     =  result.x
        result         =  result.success
        #print(f"Result: {result}")
        #print(f"Optimal steering angle: {self.state[1]}, Optimal acc: {self.state[0]}")
        car_state = self.client.getCarState()
        
        car_controls = fsds.CarControls()
        car_controls.steering = self.state[1]
        car_controls.steering = self.wrap_steer(car_controls.steering)
        
        car_controls.throttle,car_controls.brake=self.acceleration_to_throttle_brake(self.state[0])
        print("Input :",car_controls.steering,car_controls.throttle,car_controls.brake)
        self.client.setCarControls(car_controls)
        return None
    

def main(args=None):
    while True:
        node = car()
        node.state_update()
        time.sleep(0.5)
if __name__ == "__main__":
    main()