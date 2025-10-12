from scipy.optimize import minimize

import pandas as pd
import numpy as np
import csv
import time
import math

import sys
import os.path

from Communication import Communication
from PathPlanning import Waypoints
from SLAM import Localisation
from perception import Perception
from coneperception import ConePerception


cones = []
waypointlist, waypointSectors, prevWaypoints = [], [[]], []
sectorRangeThreshold = 5
sectorsBlue, sectorsYellow, prevArrCones = [[]], [[]], []
    
#with open(os.path.join(os.path.dirname(__file__), '..','conesShifted.csv'), 'r') as file:
    #csv_reader = csv.reader(file)
    #for row in csv_reader:
        #cones.append([float(row[0]),float(row[1]),row[2],row[3]])
    
with open("C:\\Users\\Arhan\\Desktop\\FMDV\\FSDS_fm\\src\\Control\\Stanley\\cones.csv", 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        cones.append([(row[0]),float(row[1]),float(row[2])])



with open(os.path.join(os.path.dirname(__file__), '..','waypointposn.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        waypointlist.append([float(row[0]), float(row[1])])

#Clearing the car pos file
with open(os.path.join(os.path.dirname(__file__),'..','carpos.csv'), 'w', newline='') as file:
    pass
file.close()

def append_values_to_csv(row):
    with open(os.path.join(os.path.dirname(__file__),'..','carpos.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("127.0.0.1")

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

# ~/basicController/small_track_shifted.csv



class Mpc(Communication):
    def __init__(self):
        super().__init__()

        self.N = 5  # horizon
        self.dt = 0.1
        self.Lf = 1.58 / 2

        # Reference values
        self.ref_v = 6.0

        # weights (same as ROS version)
        self.w_cte = 30
        self.w_epsi = 30
        self.w_v = 10
        self.w_delta = 23
        self.w_a = 5

        # use waypointlist loaded at module level
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
        #self.path = waypointlist.copy()
        self.current_waypoint_idx = 0
        self.ref_path_x = np.array([point[0] for point in self.path])
        self.ref_path_y = np.array([point[1] for point in self.path])

        # Keep old accel/steer if you want smoothing (optional)
        self.old_acceleration = 0.0
        self.old_steer = 0.0

        # CSV logging
        self.csv_file = open('mpc_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['car_x', 'car_y', 'next_point_x', 'next_point_y'])

        print("MPC constructor called")

    def advance_waypoint(self, x, y, waypoint_idx):
        path_length = len(self.path)
        while True:
            current_idx = waypoint_idx % path_length
            next_idx = (waypoint_idx + 1) % path_length

            A = self.path[(current_idx - 1) % path_length]
            B = (x, y)
            C = self.path[current_idx]

            AB = (B[0] - A[0], B[1] - A[1])
            AC = (C[0] - A[0], C[1] - A[1])

            dot_product = AB[0] * AC[0] + AB[1] * AC[1]
            AC_magnitude_squared = AC[0]**2 + AC[1]**2

            if AC_magnitude_squared == 0:
                projection_ratio = 0
            else:
                projection_ratio = dot_product / AC_magnitude_squared

            if projection_ratio > 1.0:
                waypoint_idx = next_idx
            else:
                break
        return waypoint_idx

    def next_points(self, vehicle_x, vehicle_y, vehicle_psi):
        self.current_waypoint_idx = self.advance_waypoint(
            vehicle_x, vehicle_y, self.current_waypoint_idx
        )
        return self.path[self.current_waypoint_idx]

    def run_mpc(self):
        # Ensure state is up to date (MainLoop calls state_update before run_mpc)
        # but safe to call here once if needed:
        self.state_update()

        px = self.x
        py = self.y
        psi = self.yaw
        v = self.v

        # pick target waypoint like ROS version
        ref_x, ref_y = self.next_points(px, py, psi)

        # logging CSV
        self.csv_writer.writerow([px, py, ref_x, ref_y])
        self.csv_file.flush()

        target_x = ref_x
        target_y = ref_y

        # path angle / heading error helpers
        path_angle = math.atan2(target_y - py, target_x - px)
        theta = abs(psi - path_angle)

        # normalize theta similar to ROS code
        while theta > math.pi:
            theta -= 2 * math.pi
        theta = abs(theta)
        if theta > math.pi / 2:
            theta = math.pi - theta

        dynamic_velocity = self.ref_v * (math.cos(theta)**4)

        cte = math.sqrt((px - target_x)**2 + (py - target_y)**2)
        heading_error = math.atan2(target_y - py, target_x - px) - psi

        # normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        state = [px, py, psi, v, cte, heading_error, dynamic_velocity]

        solution = self.solve_mpc(state)
        steer_value = float(solution[0])
        throttle_value = float(solution[1])

        # Log values (console)
        print(
            f"MPC -> steer: {steer_value:.4f}, accel: {throttle_value:.4f}, "
            f"cte: {cte:.4f}, v: {v:.4f}, dyn_v: {dynamic_velocity:.4f}, theta: {theta:.4f}"
        )

        # Send controls to FSDS using inherited communication method
        try:
            # Note: Communication.inputingValue expects (acceleration, steering_angle)
            self.inputingValue(throttle_value, steer_value)
        except Exception as e:
            print(f"Failed to send controls: {e}")

        return steer_value, throttle_value

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

                # update waypoint index and get reference
                local_waypoint_idx = self.advance_waypoint(x, y, local_waypoint_idx)
                ref_x, ref_y = self.path[local_waypoint_idx % len(self.path)]

                # cross-track and heading error (global-based as in ROS code)
                cte = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                epsi = math.atan2(ref_y - y, ref_x - x) - psi

                # normalize epsi
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

        bounds = [(-0.5, 0.5), (-2.0, 5.0)]
        initial_guess = [0.0, 0.8]
        result = minimize(simulate, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            return result.x
        else:
            print(f"MPC optimization failed: {result.message}")
            return np.array([0.0, 0.0])

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
