import rclpy
from scipy.optimize import minimize
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped as control
from nav_msgs.msg import Odometry as pos
from std_msgs.msg import Float64 #For car steering

import csv
import time
import numpy as np

class car(Node):
    def __init__(self):
        super().__init__("MPC_Controller")
        self.waypoints = list()
        with open("/home/gitaansh/FM/basicController/src/MPC/waypointposn.csv","r") as f:
            r = csv.reader(f)
            for i in r:
                self.waypoints.append([float(i[0]),float(i[1])])
        
        #path data
        self.m         =  self.m_perp = self.theta =  self.prev_op_sign      =  self.wp_index  =  0
        self.theta_MPC =  self.m_perp_MPC  =  self.prev_op_MPC = self.yaw_MPC  =   0  #MPC prediction data
        
        #lap timer data
        self.lap_time_start =  None  
        self.total_laps     =  0 

        #car data 
        self.state      = [1,0] 
        self.x          = self.y  =  self.yaw = 0
        self.vel        = [0,0]
        self.WheelBase  = 1.580 #m

        #ROS2 objects
        self.publisher = self.create_publisher(control, '/cmd', 10)
        self.car       = control()
        self.pos       = self.create_subscription(pos,'ground_truth/odom',self.state_update,10)
        self.publisher.publish(self.car)
        self.subsciber_carSteer = self.create_subscription(Float64, '/ground_truth/car_steer', self.updateCarSteer, 10)

        self.currentSteer = 0.0

    def updateCarSteer(self, Message):
        self.currentSteer = Message.data

    def state_update(self,msg): #callback function for updating the state of the car
       
        q          =   msg.pose.pose.orientation #quarternion data
        t3         =   +2.0 * (q.w * q.z + q.x * q.y)
        t4         =   +1.0 - 2.0 * (q.y*q.y + q.z * q.z)
        self.yaw   =  np.arctan2(t3, t4)
        self.vel   =   msg.twist.twist.linear
        self.theta =   self.straight_line(self.wp_index)
        self.x     =  msg.pose.pose.position.x
        self.y     =  msg.pose.pose.position.y
        
        self.update_waypoint(self.x,self.y)
        self.optimize_control()
        return None
        
    def global_to_car_frame(self,x,y,yaw,wp_index): #corrected
          rotation_matrix = np.array([
                                        [np.cos(-yaw), -np.sin(-yaw)],
                                        [np.sin(-yaw), np.cos(-yaw)]
                                    ])
        
          points   =  np.array([self.waypoints[wp_index%46][0]-x,self.waypoints[wp_index%46][1]- y])
          rotated  =  rotation_matrix @ points
          return rotated[0], rotated[1]
    
    def wrap_to_pi(self,yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi
    

    def heading_error(self,x,y,yaw,wp_index): #corrected
        x,y   = self.global_to_car_frame(x,y,yaw,wp_index)
        return abs(np.arctan2(y,x)) 
    
    def distance_to_waypoint(self,x1,y1,wp_index): #corrected
        xy = self.waypoints[(wp_index+1 )%46]
        return ((xy[0]-x1)**2 + (xy[1]-y1)**2)**0.5

    def delta_steer(self,steer):
        return abs(steer-self.state[1])
         
    def update_waypoint(self,x,y):              #function for lap timer and waypoint index updation
        x1 = self.waypoints[self.wp_index%46][0]
        y1 = self.waypoints[self.wp_index%46][1] #checked once
        c  =  y1-((self.m_perp)*x1)
        op = self.m_perp*x + c - y
        if (self.prev_op_sign>0 and op<0) or (self.prev_op_sign<0 and op>0) or op==0 :
                self.wp_index+=1
                self.straight_line(self.wp_index)
                if self.wp_index%46 == 1:
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
        x0  =   self.waypoints[(wp_index-k)%46][0]
        y0  =   self.waypoints[(wp_index-k)%46][1]       
        x1  =   self.waypoints[(wp_index-0)%46][0]
        y1  =   self.waypoints[(wp_index-0)%46][1]
        x2  =   self.waypoints[(wp_index+k)%46][0]
        y2  =   self.waypoints[(wp_index+k)%46][1]
        A   =   -(y2-y1) #Ax + By = C (straight line std form)
        B   =   (x2-x1)   
        C  =   -(y1-y0)  
        D =     (x1-x0)
        c0 = abs((D*(y0-y) - C*(x0-x))/(C**2 + D**2)**0.5)
        c1 = abs((B*(y1-y) - A*(x1-x))/(A**2 + B**2)**0.5)
        c = 0.3*c0 + 0.7*c1
        return c
    

    def straight_line(self,wp_index,x=1,k=1):
        x2  =  self.waypoints[(wp_index+k)%46][0]
        y2  =  self.waypoints[(wp_index+k)%46][1]

        x1  =  self.waypoints[wp_index%46][0]
        y1  =  self.waypoints[wp_index%46][1]
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
        x1 = self.waypoints[wp_index%46][0]
        y1 = self.waypoints[wp_index%46][1] #checked once
        c  = y1-((self.m_perp_MPC)*x1)
        op = self.m_perp_MPC*x + c - y
        if (self.prev_op_MPC>0 and op<0) or (self.prev_op_MPC<0 and op>0) or op==0 :
                wp_index+=1                     
        self.prev_op_MPC = op
        #print("predicted = ",wp_index)
        return wp_index
    
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
        deltaSteer =  state[1]
        v     =  [self.vel.x,self.vel.y]
        self.yaw_MPC = self.yaw 

        #prediction 1 
        # s =  np.clip(steer,-0.2,0.2)
        # self.yaw_MPC += s
        steer = self.currentSteer + min(0.5*1.04,np.abs(deltaSteer))*np.sign(deltaSteer)
        x,y,v1      =  self.predict(self.x,self.y,a,v,steer,0.5)
        wp_index   =  self.update_waypoint_predict(x,y,self.wp_index)
        cost1      =  self.cost_function(steer,x,y,wp_index,v)

        #prediction 2
        # s =  np.clip(steer,-0.2,0.2)
        # self.yaw_MPC += s
        steer = self.currentSteer + min(0.2*1.04,np.abs(deltaSteer))*np.sign(deltaSteer)
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
    
    def optimize_control(self):
        initial_guess  =  np.array([self.state[0],self.state[1]] )#Initial guess for scipy
        bounds = ([(-1, 1),(-0.52, 0.52)])
        result         =  minimize(self.optimize_function, initial_guess, method='slsqp',bounds=bounds) #optimizing the cost function
        self.state     =  result.x
        result         =  result.success
        #print(f"Result: {result}")
        #print(f"Optimal steering angle: {self.state[1]}, Optimal acc: {self.state[0]}")

        self.car.drive.steering_angle = self.state[1]
        self.car.drive.acceleration   = self.state[0]
        self.publisher.publish(self.car)
        return None
    

def main(args=None):
    rclpy.init(args=args)
    node = car()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == "__main__":
    main()