
import math
import numpy as np
import csv
from scipy.optimize import minimize

import pandas as pd
import numpy as np
import csv
import time
import math

import sys
import os.path

##from Communication import Communication
from PathPlanning import Waypoints
from SLAM import Localisation
from perception import Perception

"""
FSDS Requirements
"""
# Add fsds package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("")

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

# ~/basicController/small_track_shifted.csv


class MPCController():
    def __init__(self):
        carState = client.getCarState(vehicle_name='FSCar')
        carState = carState.kinematics_estimated
        self.carState = carState
        
        self.N = 5 #horizon, was 12
        self.dt = 0.1 #timestep
        self.Lf = 1.58/2  # distance from center of mass to front axle
        
        # Reference values
        self.ref_v = 3.0
        self.w_cte = 30 #was 35   #cte error ka weight
        self.w_epsi = 30   #was 35.5  #heading error ka weight
        self.w_v = 7  #was 18   #velocity weight
        self.w_delta = 30  #steering angle weight
        self.w_a = 5        #acc weight
        
        self.path = [  
            (4, 0.5), (6.1, 0.46), (8.96, 0.63), (12.9, 0.54), (16.5, 0.91), (20.14, 2.02),
            (23.03, 3.01), (25.72, 4.12), (27.05, 3.85), (29.24, 3.03), (30.23, 2.15),
            (31.74, 0.11), (32.04, -1.34), (31.85, -4.60), (30.93, -7.44), (30.10, -9.54),
            (28.42, -11.91), (26.17, -13.64), (23.26, -15.09), (20.75, -16.53),
            (16.42, -18.71), (12.72, -20.56), (9.16, -22.59), (6.34, -24.41),
            (3.44, -25.23), (2.01, -24.97), (-0.24, -23.59), (-1.13, -22.77),
            (-3.03, -20.33), (-3.38, -16.96), (-3.54, -15.76), (-3.28, -12.89),
            (-2.86, -10.3), (-2.56, -7.25), (-2.59, -4.27), (-1.47, -1.72),
            (-0.7, -0.64)
        ]

        self.current_waypoint_idx = 0
        self.ref_path_x = np.array([point[0] for point in self.path])
        self.ref_path_y = np.array([point[1] for point in self.path])
        
        # Initialize CSV file
        self.csv_file = open('mpc_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['car_x', 'car_y', 'next_point_x', 'next_point_y'])

    def car_callback(self):
        self.run_mpc()

    def advance_waypoint(self, x, y, waypoint_idx):
        path_length = len(self.path)
        while True:
            current_idx = waypoint_idx % path_length #current waypt
            next_idx = (waypoint_idx + 1) % path_length #for wrap around path loop
            
            A = self.path[(current_idx - 1) % path_length] #previous point
            B = (x, y) #current pos
            C = self.path[current_idx] #target waypt
            
            AB = (B[0] - A[0], B[1] - A[1])
            AC = (C[0] - A[0], C[1] - A[1])
            
            dot_product = AB[0] * AC[0] + AB[1] * AC[1]
            AC_magnitude_squared = AC[0]**2 + AC[1]**2
            
            #check if ratio is greater than 1 if yes then update waypt
            if AC_magnitude_squared == 0:  
                projection_ratio = 0
            else:
                projection_ratio = dot_product / AC_magnitude_squared
            
            if projection_ratio > 1.0:
                waypoint_idx = next_idx
            else:
                break
        return waypoint_idx

    def next_points(self, vehicle_x, vehicle_y):
        self.current_waypoint_idx = self.advance_waypoint(vehicle_x, vehicle_y, self.current_waypoint_idx)
        return self.path[self.current_waypoint_idx]

    def run_mpc(self):
        drive_msg = client.getCarState(vehicle_name='FSCar')
        drive_msg = drive_msg.kinematics_estimated
        self.carState = drive_msg
        self.x = drive_msg.position.x_val
        self.y = drive_msg.position.y_val
        q = self.carState.orientation

        
        t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        t4 = +1.0 - 2.0 * (q.y_val*q.y_val + q.z_val* q.z_val)
        self.yaw = -np.arctan2(t3, t4)
        print("Yaw ",self.yaw)

        
        self.vel = [float(drive_msg.linear_velocity.x_val), float(drive_msg.linear_velocity.y_val)]
        self.v_x = self.vel[0]
        self.v_y = self.vel[1]
        self.v = (self.v_x**2 + self.v_y**2)**0.5

        ref_x, ref_y = self.next_points(self.x, self.y) #target waypt

        # Write to CSV file
        self.csv_writer.writerow([self.x, self.y, ref_x, ref_y])
        self.csv_file.flush()

        target_x = ref_x
        target_y = ref_y

        # Calculate theta (angle between car heading and path direction) #to calculate heading error
        path_angle = math.atan2(target_y - self.y, target_x - self.x) 
        theta = abs(self.yaw - path_angle)
        
        # normalize angle
        while theta > math.pi:
            theta -= 2 * math.pi
        theta = abs(theta)
        if theta > math.pi/2:
            theta = math.pi - theta
            
        dynamic_velocity=self.ref_v*((math.cos(theta)**(4/1)))  #use costheta for descreasing speed when theta increases

        cte = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)
        heading_error = math.atan2(target_y - self.y, target_x - self.x) - self.yaw

        #changes made here
        while heading_error > math.pi:
            heading_error -= 2*math.pi
        while heading_error < math.pi:
            heading_error += 2 * math.pi

        state = [self.x, self.y, self.yaw, self.v, cte, heading_error, dynamic_velocity]

        solution = self.solve_mpc(state)

        steer_value = solution[0]
        throttle_value = solution[1]

        drive_msg = fsds.CarControls()
        drive_msg.steering = -1*steer_value
        drive_msg.throttle = throttle_value
        print("Throttle:",throttle_value)
        print("Steer Value:",steer_value)
        client.setCarControls(drive_msg)

    def solve_mpc(self, state):
        x0, y0, psi0, v0, cte0, epsi0, dynamic_v = state
        N = self.N
        dt = self.dt
        Lf = self.Lf
        
        initial_waypoint_idx = self.current_waypoint_idx

        def simulate(u):
            x, y, psi, v = x0, y0, psi0, v0
            cost = 0.0
            local_waypoint_idx = initial_waypoint_idx

            for i in range(N):
                delta = u[0]
                a = u[1]

                x += v * math.cos(psi) * dt
                y += v * math.sin(psi) * dt
                psi += v * delta / Lf * dt
                v += a * dt

                #checking waypoint logic
                local_waypoint_idx = self.advance_waypoint(x, y, local_waypoint_idx)
                ref_x, ref_y = self.path[local_waypoint_idx % len(self.path)]
 
                cte = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                epsi = math.atan2(ref_y - y, ref_x - x) - psi

                while epsi > math.pi:
                    epsi -= 2 * math.pi
                while epsi < -math.pi:
                    epsi += 2 * math.pi

                cost += self.w_cte * cte**2
                cost += self.w_epsi * epsi**2
                cost += self.w_v * (v - dynamic_v)**2
                cost += self.w_delta * delta**2
                cost += self.w_a * a**2

            return cost

        bounds = [(-0.5, 0.5), (-2.0, 5.0)]    #was 3.0 # steering and throttle
        initial_guess = [0.0, 0.8]   #was 0.1
        result = minimize(simulate, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            return result.x
        else:
            return [0.0, 0.0]

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
timerPeriod = 1/30


def main(args=None):
    obj = MPCController()
    start = time.time()
    try:
        while True:
            dt = time.time() - start
            if dt >= timerPeriod:
                obj.car_callback()
                start = time.time()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
        client.enableApiControl(True)
    except Exception as e:
        print(f"Error encountered: {e}")
        client.reset()
        client.enableApiControl(True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
