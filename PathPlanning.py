import pandas as pd
import numpy as np
import time
import os.path

waypointsDf = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wpsNew.csv')))

class Waypoints :
    def __init__(self):
        self.currentWaypoint = 0
        self.prev_flag = -1
        self.v_max = 3
        self.lap_start = time.time()
        self.v_sum, self.v_count = 0, 0

        #Only for the monitor, its a list of x,y of each waypoint. did it here to avoid adding waypoints in the function of car
        self.waypoints = []
        for i in range(1, len(waypointsDf)):
            self.waypoints.append(list(waypointsDf.iloc[i,:]))
    
    def GetRefVel(self, car_pos, power):

    
        num_points = len(waypointsDf)
        

    # Get x, y for current and next waypoints
        x1, y1 = waypointsDf.iloc[self.currentWaypoint % num_points, 0], waypointsDf.iloc[self.currentWaypoint % num_points, 1]
        x2, y2 = waypointsDf.iloc[(self.currentWaypoint+1) % num_points, 0], waypointsDf.iloc[(self.currentWaypoint+1) % num_points, 1]
        x3, y3 = waypointsDf.iloc[(self.currentWaypoint+2) % num_points, 0], waypointsDf.iloc[(self.currentWaypoint+2) % num_points, 1]

    # Compute angles
        theta1 = np.arctan2((car_pos[1] - y1) , (car_pos[0] - x1))
        theta2 = np.arctan2((car_pos[1] - y2) , (car_pos[0] - x2))
        theta3 = np.arctan2((car_pos[1] - y3) , (car_pos[0] - x3))

    # Weighted average
        theta = 0.7*theta1 + 0.2*theta2 + 0.1*theta3

    # Reference velocity
        v_ref = np.abs(self.v_max * np.cos(theta)**(1/power))
        return v_ref


    def IfCrossedWaypoint(self, car_x, car_y, update = True):
        
        X = list(waypointsDf.iloc[:,0])
        Y = list(waypointsDf.iloc[:,1])
        num_points = len(waypointsDf)
        
        delX = X[(self.currentWaypoint+1)%num_points] - X[self.currentWaypoint%num_points]
        delY = Y[(self.currentWaypoint+1)%num_points] - Y[self.currentWaypoint%num_points]

        flag = delX*car_x + delY*car_y - delX*X[self.currentWaypoint%num_points] - delY*Y[self.currentWaypoint%num_points] # equation of straight line perpendicular to line of way points

        #made a crossed here instead of directly checking prev flag * flag and then updated prev flag 
        crossed = self.prev_flag * flag 
        self.prev_flag = flag
        if crossed and update:
            if (self.currentWaypoint+1)%num_points == 0:
                self.lap_end = time.time()
                print("Lap time - ",self.lap_end - self.lap_start)
                #print("Average lap velocity = ", self.v_sum/self.v_count)
                self.v_sum = 0
                self.v_count = 0
                self.lap_start = time.time()
            self.currentWaypoint = (self.currentWaypoint + 1)%num_points
        if update is False:
            return True
            #print("current waypoint : ",self.currentWaypoint%37)
    
    def GetError(self, x, y, yaw):
        num_points = len(waypointsDf)
        X = list(waypointsDf.iloc[:,0])
        Y = list(waypointsDf.iloc[:,1])
        currentWaypoint = (self.currentWaypoint % num_points)
        if self.IfCrossedWaypoint(x, y, False):
            currentWaypoint = (self.currentWaypoint + 1)%num_points

        
            delX = X[currentWaypoint%num_points] - X[(currentWaypoint-1)%num_points]
            delY = Y[currentWaypoint%num_points] - Y[(currentWaypoint-1)%num_points]
        
            delX = X[currentWaypoint%num_points] - 0
            delY = Y[currentWaypoint%num_points] - 0

        crosstrack_error = (delX*(Y[(currentWaypoint-1)%num_points]-y) - delY*(X[(currentWaypoint-1)%num_points]-x))/(delX**2 + delY**2)**0.5

        global_waypoint=np.array([X[currentWaypoint%num_points], Y[currentWaypoint%num_points]])
        car_posn=np.array([x, y])
        relative_posn=-car_posn+global_waypoint
        rotation_matrix=np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)]])
        
        waypoint_posn_in_car=rotation_matrix @ relative_posn
        
        waypoint_heading = np.arctan2(waypoint_posn_in_car[1], waypoint_posn_in_car[0])
        heading_error = waypoint_heading 

        '''waypoint_heading = np.arctan2(delY,delX)
        heading_error =  waypoint_heading - yaw'''
        return crosstrack_error, heading_error
    
    def CalcAvgVel(self, velocity):
        self.v_sum += velocity
        self.v_count += 1