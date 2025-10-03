import sys
import os
import csv
import numpy as np

#from rclpy.node import Node     ONLY FOR TIMER IN MAIN


"""
FSDS Requirements
"""
# Add fsds package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("127.0.0.1")

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

# ~/basicController/small_track_shifted.csv

class Communication():
    def __init__(self):
        self.x, self.y, self.z = 0,0,0
        self.v, self.v_x, self.v_y, self.v_z = 0,0,0,0        
        self.vel = [0,0]        
        self.yaw = 0
        self.carState = None
        self.waypoint_sign = -1
        self.xCheck = [0,0,0] 
        self.maxVel = 5
        self.minVel = 3
        self.lk = []
        self.prevlk = []
        
        self.sectorsBlue, self.sectorsYellow, self.prevArrCones = [[]], [[]], []


        #lap timer data
        self.lap_time_start =  None  
        self.total_laps = 0 

        #car data 
        self.WheelBase = 1.580 #m
    
    def state_update(self): #callback function for updating the state of the car
        
        carState = client.getCarState(vehicle_name='FSCar')
        carState = carState.kinematics_estimated
        self.carState = carState

        self.vel = [float(carState.linear_velocity.x_val), float(carState.linear_velocity.y_val)]
        self.v_x = self.vel[0]
        self.v_y = self.vel[1]
        self.v = (self.v_x**2 + self.v_y**2)**0.5
        
        
        q = self.carState.orientation

        t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        t4 = +1.0 - 2.0 * (q.y_val*q.y_val + q.z_val* q.z_val)
        self.yaw = np.arctan2(t3, t4)
        print("Yaw ",self.yaw)

        self.x = carState.position.x_val
        self.y = carState.position.y_val
        self.xCheck = [self.x, self.y, self.yaw]
        
        #commenting shit for now
        
        #print("X, Y ",self.x,self.y)

        #print("Current Accel ",carState.linear_acceleration)



        return None

    def inputingValue(self, acceleration, steering_angle):
        car_controls = fsds.CarControls()
        #acceleration = -acceleration
        throttle, brake = self.acceleration_to_throttle_brake(acceleration)
        #print(f"Inputted throttle : {throttle} brake : {brake}")
        car_controls.steering = -steering_angle
        car_controls.throttle = throttle
        car_controls.brake = brake
        client.setCarControls(car_controls)


    def acceleration_to_throttle_brake(self,acc, a_max=1.0, b_max=1.0):
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
            #brake = 1
        return np.clip(throttle, 0.0, 0.2), np.clip(brake, 0.0, 1.0)