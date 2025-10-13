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

        theta1 = np.arctan((car_pos[1] - waypointsDf.iloc[(self.currentWaypoint%37)][1])/(car_pos[0] - waypointsDf.iloc[(self.currentWaypoint%37)][0]))
        theta2 = np.arctan((car_pos[1] - waypointsDf.iloc[(self.currentWaypoint+1)%37,:][1])/(car_pos[0] - waypointsDf.iloc[(self.currentWaypoint+1)%37,:][0]))
        theta3 = np.arctan((car_pos[1] - waypointsDf.iloc[(self.currentWaypoint+2)%37,:][1])/(car_pos[0] - waypointsDf.iloc[(self.currentWaypoint+2)%37,:][0]))

        theta = 0.7*theta1 + 0.2*theta2 +0.1*theta3
        v_ref = np.abs(self.v_max*np.cos(theta)**(1/power))
        return v_ref

    def IfCrossedWaypoint(self, car_x, car_y, update = True):
        X = list(waypointsDf.iloc[:,0])
        Y = list(waypointsDf.iloc[:,1])
        delX = X[(self.currentWaypoint+1)%37] - X[self.currentWaypoint%37]
        delY = Y[(self.currentWaypoint+1)%37] - Y[self.currentWaypoint%37]

        flag = delX*car_x + delY*car_y - delX*X[self.currentWaypoint%37] - delY*Y[self.currentWaypoint%37] # equation of straight line perpendicular to line of way points

        if self.prev_flag*flag < 0 and update is True:
            if (self.currentWaypoint+1)%37 == 0:
                self.lap_end = time.time()
                print("Lap time - ",self.lap_end - self.lap_start)
                #print("Average lap velocity = ", self.v_sum/self.v_count)
                self.v_sum = 0
                self.v_count = 0
                self.lap_start = time.time()
            self.currentWaypoint+=1
        if update is False:
            return True
            #print("current waypoint : ",self.currentWaypoint%37)
    
    def GetError(self, x, y, yaw):
        X = list(waypointsDf.iloc[:,0])
        Y = list(waypointsDf.iloc[:,1])
        currentWaypoint = self.currentWaypoint
        if self.IfCrossedWaypoint(x, y, False):
            currentWaypoint = self.currentWaypoint + 1

        try:
            delX = X[currentWaypoint%37] - X[(currentWaypoint-1)%37]
            delY = Y[currentWaypoint%37] - Y[(currentWaypoint-1)%37]
        except:
            delX = X[currentWaypoint%37] - 0
            delY = Y[currentWaypoint%37] - 0

        crosstrack_error = (delX*(Y[(currentWaypoint-1)%37]-y) - delY*(X[(currentWaypoint-1)%37]-x))/(delX**2 + delY**2)**0.5

        global_waypoint=np.array([X[currentWaypoint%37], Y[currentWaypoint%37]])
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