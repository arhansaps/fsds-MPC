from scipy.optimize import minimize, Bounds
import pandas as pd
import numpy as np
import csv
import time
import sys
import os.path


from Communication import Communication
from PathPlanning import Waypoints
from SLAM import Localisation
from perception import Perception

# Load cones data
cones = []
with open(os.path.join("C:\\Users\\Arhan\\Desktop\\FMDV\\FSDS_fm\\src\\Control\\conesShifted.csv"), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        cones.append([float(row[0]),float(row[1]),row[2],row[3]])

# Clear car position file
with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'w', newline='') as file:
    pass

def append_values_to_csv(row):
    with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# EXACT constants from your working min_curv.py
KP = 2
KI = 4  
KD = 1
WHEELBASE = 1.530
MAXSTEER = 0.52
TIMERVALUE = 0.02

# FSDS client setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("127.0.0.1")
client.confirmConnection()
client.enableApiControl(True)

class Mpc(Communication):
    def __init__(self):
        print("MPC constructor called")
        super().__init__()
        
        # EXACT values from your working code
        self.maxVel = 5

        self.minVel = 3
        
        # Variables from your working code
        self.previous_result = None
        self.ok = True
        self.ind = 0
        
        # PID controller
        self.integral = 0
        self.prev_error = 0
        
        # Objects
        self.waypoints = Waypoints()
        self.localisation = Localisation()
        self.perception = Perception()
        self.localisation.CreateMap(cones)

    def pidControl(self, targetVel):
        """EXACT PID from your working code"""
        v_error = targetVel - self.v
        self.integral += v_error * TIMERVALUE
        a = KP * v_error + KI * self.integral + KD * (v_error - self.prev_error) / TIMERVALUE
        self.prev_error = v_error
        return a

    def euclidean_dist(self, point1, point2):
        """From your working code"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    """def rotate_point(self, xCheck, cone): 
        EXACT from your working code
        theta = xCheck[2]
        rotationMat = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]], dtype="float")
        newCone = rotationMat.dot(cone[:2])
        newCone = newCone.flatten()
        return newCone[0], newCone[1]"""
    
    # modified rotate point
    def rotate_point(self, xCheck, point):
        theta = xCheck[2]
        dx = point[0] - xCheck[0]
        dy = point[1] - xCheck[1]
    
        # vehicle frame
        x_veh = dx * np.cos(theta) + dy * np.sin(theta)
        y_veh = -dx * np.sin(theta) + dy * np.cos(theta)
    
        return x_veh, y_veh
    
    #added new function, og predict motion was some bullshit idk
    def predict_motion(self, x_current, velocity, steering, dt):

        x, y, theta = x_current
        
        # calculate heading rate
        if abs(steering) < 1e-6:
            theta_new = theta
            x_new = x + velocity * dt * np.cos(theta)
            y_new = y + velocity * dt * np.sin(theta)
        else:
            
            theta_dot = velocity * np.tan(steering) / WHEELBASE
            theta_new = theta + theta_dot * dt
            
            # updated position using avg
            avg_theta = (theta + theta_new) / 2.0
            x_new = x + velocity * dt * np.cos(avg_theta)
            y_new = y + velocity * dt * np.sin(avg_theta)
    
        
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        return [x_new, y_new, theta_new]


    def getWaypointSign(self, xCheck, prevWaypoint, currWaypoint):
        """From your working code"""
        slopeWaypoints = (currWaypoint[1] - prevWaypoint[1]) / (currWaypoint[0] - prevWaypoint[0] + 1e-6)
        perpLineSlope = -1 / slopeWaypoints
        a, b, c = perpLineSlope, -1, currWaypoint[1] - perpLineSlope * currWaypoint[0]
        temp = a * xCheck[0] + b * xCheck[1] + c
        currSign = temp / (abs(temp) + 1e-6)
        return currSign

    def checkIfWaypointCrossed(self, xCheck, prevWaypoint, currWaypoint, waypointSign):
        """From your working code"""
        currSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
        if (currSign * waypointSign) < 0:
            return True
        return False

    def cte_function(self, x):
        """SIMPLIFIED: Work in global frame like your working code"""
        # Start from current position in global frame
        xCheck = [self.x, self.y, self.yaw]
        cost = 0
        
        steerSign = np.sign(x[1])
        waypointSign = self.waypointSign
        DT = 0.1
        ind = self.ind
        
        if ind >= len(self.waypoints.waypoints):
            return 1000
            
        currWaypoint = self.waypoints.waypoints[ind]
        prevWaypoint = self.waypoints.waypoints[ind - 1]
        alpha = 0.3
        carSteer = 0

        for i in range(5):

            #making changes here

            #carSteer += min(DT * np.pi, abs(x[1])) * steerSign
            #dt was added twice here hence calculations were very small
            steering = x[1]
            velocity = x[0]

            xCheck = self.predict_motion(xCheck, velocity, steering, DT)

            """SIMPLIFIED: Global frame motion model
            xCheck[2] += carSteer * DT  # Update heading
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])  # Update x
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])  # Update y"""

            if (self.checkIfWaypointCrossed(xCheck, prevWaypoint, currWaypoint, waypointSign)):
                ind = (ind + 1) % len(self.waypoints.waypoints)
                currWaypoint = self.waypoints.waypoints[ind]
                prevWaypoint = self.waypoints.waypoints[ind - 1]
                waypointSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
            
            # Distance to line between waypoints (GLOBAL FRAME)
            slope = (currWaypoint[1] - prevWaypoint[1]) / (currWaypoint[0] - prevWaypoint[0] + 1e-6)
            a, b, c = slope, -1, prevWaypoint[1] - slope * prevWaypoint[0]
            perp_dist = abs(a * xCheck[0] + b * xCheck[1] + c) / np.sqrt(a**2 + b**2)
            cost += (1 - alpha) * alpha**i * perp_dist
        
        return cost

    def dist_from_waypoint_function(self, x):
        """SIMPLIFIED: Distance to waypoints in global frame"""
        xCheck = [self.x, self.y, self.yaw]
        cost = 0
        steerSign = np.sign(x[1])
        waypointSign = self.waypointSign
        DT = 0.1
        ind = self.ind
        
        if ind >= len(self.waypoints.waypoints):
            return 1000
            
        currWaypoint = self.waypoints.waypoints[ind]
        prevWaypoint = self.waypoints.waypoints[ind - 1]
        alpha = 0.5
        carSteer = 0

        for i in range(5):  # Reduced from 10 to 5
            #carSteer += min(DT * np.pi, abs(x[1])) * steerSign

            steering = x[1]
            velocity = x[0]
            
            """SIMPLIFIED: Global frame motion
            xCheck[2] += carSteer * DT
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])"""

            xCheck = self.predict_motion(xCheck, velocity, steering, DT)


            if (self.checkIfWaypointCrossed(xCheck, prevWaypoint, currWaypoint, waypointSign)):
                ind = (ind + 1) % len(self.waypoints.waypoints)
                currWaypoint = self.waypoints.waypoints[ind]
                prevWaypoint = self.waypoints.waypoints[ind - 1]
                waypointSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
            
            # GLOBAL FRAME: Direct euclidean distance
            cost += (1 - alpha) * alpha**i * self.euclidean_dist(xCheck[:2], currWaypoint)
        
        return cost

    def delta_steer_function(self, x):
        """Steering smoothness"""
        return abs(x[1] - 0)

    """def theta_next_function(self, x):
        SIMPLIFIED: Heading error to next waypoint (car frame)
        if self.ind >= len(self.waypoints.waypoints):
            return 1000
            
        nextWaypoint = self.waypoints.waypoints[(self.ind + 1) % len(self.waypoints.waypoints)]
        
        # Transform waypoint to car frame for heading calculation
        waypointCarFrame = self.rotate_point(
            [0, 0, 2 * np.pi - self.yaw], 
            [nextWaypoint[0] - self.x, nextWaypoint[1] - self.y, 0]
        )
        thetaTemp = abs(np.arctan2(waypointCarFrame[1], waypointCarFrame[0]))
        
        return thetaTemp"""
    
    def theta_next_function(self, x):
        """FIXED: Heading error to next waypoint"""
        if self.ind >= len(self.waypoints.waypoints):
            return 1000
            
        nextWaypoint = self.waypoints.waypoints[(self.ind + 1) % len(self.waypoints.waypoints)]
        
        #heading
        dx = nextWaypoint[0] - self.x
        dy = nextWaypoint[1] - self.y
        desired_heading = np.arctan2(dy, dx)
    
        heading_error = desired_heading - self.yaw
        
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        return abs(heading_error)

    def mpc_cost_function(self, x):
        """EXACT cost function from your working code"""
        return (4 * self.cte_function(x) + 
                10 * self.dist_from_waypoint_function(x) + 
                1.5 * self.delta_steer_function(x) + 
                250 * self.theta_next_function(x))

    def Calculate(self):
        """SIMPLIFIED: Calculate control"""
        try:
            
            curr_pos = [self.x, self.y]
            min_dist = float('inf')
            
            #made changes here, inconsistency between yaw idk why 360-yaw was added i added yaw normally
            for i, waypoint in enumerate(self.waypoints.waypoints):
                dist = self.euclidean_dist(curr_pos, waypoint)
                waypoint_car_frame = self.rotate_point(
                    #[0, 0, 2 * np.pi - self.yaw],
                    [self.x,self.y,self.yaw]
                    #[waypoint[0] - curr_pos[0], waypoint[1] - curr_pos[1]]
                    ,waypoint
                )
                if waypoint_car_frame[0] > 0 and dist < min_dist:
                    min_dist = dist
                    self.ind = i

            #sign
            if len(self.waypoints.waypoints) > 1:
                currWaypoint = self.waypoints.waypoints[self.ind]
                prevWaypoint = self.waypoints.waypoints[self.ind - 1]
                self.waypointSign = self.getWaypointSign(
                    [curr_pos[0], curr_pos[1], self.yaw],
                    prevWaypoint,
                    currWaypoint
                )

            
            initial_guess = np.array([3, 0])
            bounds = Bounds([2, -MAXSTEER], [self.maxVel, MAXSTEER])  

            if self.previous_result is None:
                self.previous_result = np.array([3, 0])

            if self.ok:
                initial_guess = self.previous_result
                result = minimize(self.mpc_cost_function, initial_guess, method='SLSQP', bounds=bounds)
                self.previous_result = np.array(result.x) if result.success else self.previous_result
            else:
                bounds = Bounds([1, -MAXSTEER], [2, MAXSTEER])
                initial_guess = np.array([1.5, 0])
                result = minimize(self.mpc_cost_function, initial_guess, method='SLSQP', bounds=bounds)

            if not result.success:
                print("use previous")
                desiredVelocity = self.previous_result[0]
                steerAngle = self.previous_result[1]
            else:
                desiredVelocity = result.x[0]
                steerAngle = result.x[1]

            #steering limits
            steerAngle = np.clip(steerAngle, -MAXSTEER, MAXSTEER)
            
            #using pid control
            accel = self.pidControl(desiredVelocity)
            
            print(f"Target Vel: {desiredVelocity:.2f}, Steer: {steerAngle:.3f}, Accel: {accel:.3f}")
            
            return accel, steerAngle
            
        except Exception as e:
            print(f"MPC Calculate error: {e}")
            return 0.5, 0.0

    def MainLoop(self):
        """Main control loop"""
        try:
            self.state_update()
            self.waypoints.IfCrossedWaypoint(self.x, self.y) 
            
            acceleration, steering_angle = self.Calculate()
            self.inputingValue(acceleration, steering_angle)            
                        
            append_values_to_csv([self.x, self.y])
        except Exception as e:
            print(f"MainLoop error: {e}")
            self.inputingValue(0.1, 0.0)

timerPeriod = 1/30
        
def main(args=None):
    
    
    obj = Mpc()
    start = time.time()
    try:
        while True:
            dt = time.time() - start
            if dt >= timerPeriod:
                obj.MainLoop()
                start = time.time()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
        client.enableApiControl(True)
        obj.destroy_node()  
        
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