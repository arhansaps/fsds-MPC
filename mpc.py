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

        self.N = 5 #horizon, was 12
        self.dt = 0.1 #timestep
        self.Lf = 1.58/2  # distance from center of mass to front axle

        self.communication = Communication()
        self.waypoints = Waypoints()
        self.localisation = Localisation()
        self.perception = Perception()
        self.coneperception = ConePerception()


        self.localisation.CreateMap(cones)
        
        # Reference values
        self.ref_v = 6.0
        
        self.w_cte = 30 #was 35   #cte error ka weight
        self.w_epsi = 30   #was 35.5  #heading error ka weight
        self.w_v = 10  #was 18   #velocity weight
        self.w_delta = 23  #steering angle weight
        self.w_a = 5        #acc weight
        
        self.current_waypoint_idx = 0
        #self.ref_path_x = np.array([point[0] for point in waypointlist])
        #self.ref_path_y = np.array([point[1] for point in waypointlist])
        
        # Initialize CSV file
        self.csv_file = open('mpc_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['car_x', 'car_y', 'next_point_x', 'next_point_y'])
        
        print("MPC constructor called")

    def advance_waypoint(self, x, y, waypoint_idx):
        path_length = len(waypointlist)
        while True:
            current_idx = waypoint_idx % path_length #current waypt
            next_idx = (waypoint_idx + 1) % path_length #for wrap around path loop
            
            A = waypointlist[(current_idx - 1) % path_length] #previous point
            B = (x, y) #current pos
            C = waypointlist[current_idx] #target waypt
            
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

    def next_points(self, vehicle_x, vehicle_y, vehicle_psi):
        self.current_waypoint_idx = self.advance_waypoint(vehicle_x, vehicle_y, self.current_waypoint_idx)
        return waypointlist[self.current_waypoint_idx]

    def run_mpc(self):
        self.state_update()
        px = self.x
        py = self.y
        psi = self.yaw
        v = self.v    

        # get target waypoint
        ref_x, ref_y = self.next_points(px, py, psi)

        # logging
        self.csv_writer.writerow([px, py, ref_x, ref_y])
        self.csv_file.flush()

        # path angle / heading error helpers
        target_x = ref_x
        target_y = ref_y
        path_angle = math.atan2(target_y - py, target_x - px)
        theta = abs(psi - path_angle)

        # normalize theta
        while theta > math.pi:
            theta -= 2 * math.pi
        theta = abs(theta)
        if theta > math.pi / 2:
            theta = math.pi - theta

        dynamic_velocity = self.ref_v * (math.cos(theta) ** 4)

        cte = math.sqrt((px - target_x)**2 + (py - target_y)**2)
        heading_error = math.atan2(target_y - py, target_x - px) - psi

        # normalize heading_error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        state = [px, py, psi, v, cte, heading_error, dynamic_velocity]

        # solve MPC
        solution = self.solve_mpc(state)

        steer_value = float(solution[0])
        throttle_value = float(solution[1])

        # Log values
        print(f"steer: {steer_value:.4f}, accel: {throttle_value:.4f}, cte: {cte:.4f}, v: {v:.4f}, dyn_v: {dynamic_velocity:.4f}, theta: {theta:.4f}")

        # send controls to simulator
        # Use self.inputingValue since Mpc inherits Communication
        try:
            self.inputingValue(throttle_value, steer_value)
        except Exception as e:
            print(f"Failed to send controls: {e}")

        # return them too in case calling code wants them
        return steer_value, throttle_value

        
        #elf.get_logger().info(f"MPC: steering={steer_value:.3f}, throttle={throttle_value:.3f}, cte={cte:.3f}, v={v:.3f}, dynamic_v={dynamic_velocity:.3f}, angle={theta:.3f}")

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
                ref_x, ref_y = waypointlist[local_waypoint_idx % len(waypointlist)]

                cte = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                epsi = math.atan2(ref_y - y, ref_x - x) - psi  

                #crosstrack_error, epsi = self.waypoints.GetError(x, y, psi)


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
            #self.get_logger().warn(f"MPC optimization failed: {result.message}")
            return [0.0, 0.0]

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
    
    def MainLoop(self):
        self.state_update()
        self.waypoints.IfCrossedWaypoint(self.x, self.y)
        
        
        self.run_mpc()

        # Append current car position to CSV for logging
        append_values_to_csv([self.x, self.y])



timerPeriod = 1/30
        
def main(args=None):
    # rclpy.init(args=args)
    obj = Mpc()
    #start = time.time()
    next_call = time.time()
    try:
        while True:

    # performing cone detection in each frame method doesnt work otherwise
            obj.coneperception.perform_cone_detection()

            obj.MainLoop()
            next_call += timerPeriod
            sleep_time = max(0, next_call - time.time())
            time.sleep(sleep_time)

            
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
        client.enableApiControl(True)
    except Exception as e:
        print(f"Error encountered: {e}")
        client.reset()
        client.enableApiControl(True)
    

    #obj.show_monitor = False
    
    # rclpy.spin(obj)

    # obj.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
