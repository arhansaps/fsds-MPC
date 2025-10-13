import numpy as np
import rclpy
from rclpy.node import Node
import csv
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from eufs_msgs.msg import CarState, ConeArrayWithCovariance
from std_msgs.msg import Float64
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.optimize import Bounds
from copy import deepcopy
from fmdv_msgs.msg import MonitorData, VehicleState, ControlCommand, ConesList, PointsList, Cone
from geometry_msgs.msg import Point
from scipy.interpolate import splprep, splev
import threading
from queue import Queue



KP = 2 # Constant for proportional control
KI = 4
KD = 1
WHEELBASE = 1.530
KSTEER = 0.2 # Constant for steering control
KSOFTENING = 1 # Constant for softening control
MAXSTEER = 0.52 # Maximum steering angle
DISTOFSMALLTRACK = 101.1 # Distance of small track in meters

TIMERVALUE = 0.02

PACKAGENAME = "fmdv_code"

class Controls(Node):
    def __init__(self):
        super().__init__("controls")

        

        self.maxVel = 5
        self.minVel = 3

        self.waypointSign = -1
        self.previous_result = None

        self.xCheck = [0, 0, 0]
        self.v, self.roll, self.pitch, self.yaw, self.ind, self.angVel, self.accel, self.lk, self.prevlk, self.carSteer = 0, 0, 0, 0, 0, 0, 0, [], [], 0

        self.create_subscription(Odometry, "/ground_truth/odom", self.updateOdom, 10)
        self.controlsPub = self.create_publisher(AckermannDriveStamped, "/cmd", 10)
        self.monitor_pub = self.create_publisher(MonitorData, "/fmdv/monitor_data", 10)

        self.create_subscription(CarState, "/ground_truth/state", self.updateCarState, 10)
        self.create_subscription(ConeArrayWithCovariance, "ground_truth/cones", self.updateCones, 10)
        self.create_subscription(Float64, "ground_truth/car_steer", self.updateCarSteer, 10)

        self.waypoints, self.waypointSectors, self.prevWaypoints = [], [[]], []
        self.sectorRangeThreshold = 5
        self.sectorsBlue, self.sectorsYellow, self.prevArrCones = [[]], [[]], []

        self.timeStart = time.time()
        self.started = False # for lap start 
        self.lapEndClose = False
        self.laps = 1

        #For Minimum Curvature Path
        self.optimizer = PathOptimizer()
        self.optimization_requested = False
        self.optimization_in_progress = False
        self.optimization_complete = False
        self.optimizer_status = True

        self.ok = True # for choosing the mpc function (normal vs alternate)

        # FOR PID

        self.integral = 0
        self.prev_error = 0

        self.loopSetter()
    
    def euler_from_quaternion(self, x, y, z, w):
        """
            Converts quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
        """

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def updateCarState(self, data):
        """Updates the car state"""

        self.accel = np.sqrt(data.linear_acceleration.x ** 2 + data.linear_acceleration.y ** 2)

    def updateCarSteer(self, data):

        self.carSteer = data.data

    def updateOdom(self, data):
        """Updates the orientation of the car using the IMU"""

        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)
        self.yaw = self.wrap_to_pi(self.yaw)
        self.xCheck = [data.pose.pose.position.x, data.pose.pose.position.y, self.yaw]
        self.v = np.sqrt(data.twist.twist.linear.x ** 2 + data.twist.twist.linear.y ** 2)
        self.angVel = data.twist.twist.angular.z

    def updateCones(self,data):
        """Updates the cone locations in the car frame as seen by the camera"""

        lk = []
        for i in data.blue_cones:
            lk.append([i.point.x,i.point.y,"Blue", "Left"])
        for i in data.yellow_cones:
            lk.append([i.point.x,i.point.y,"Yellow", "Right"])
        for i in data.big_orange_cones:
            if (i.point.y < 0):
                lk.append([i.point.x,i.point.y,"Big Orange", "Right"])
            else:
                lk.append([i.point.x,i.point.y,"Big Orange", "Left"])
        
        if (lk == self.prevlk):
            return
        self.prevlk = self.lk
        self.lk = lk

    def wrap_to_pi(self, x):
        """
            Wraps the angle in the range [-pi, pi]

            Args: 
                x : Angle in radians

            Returns:
                x : Angle in radians in the range [-pi, pi]
        """

        return (x + np.pi) % (2 * np.pi) - np.pi

    def euclidean_dist(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def pControl(self, theta, currVel):
        """
            PID control to control the velocity of the car

            Args:
                theta : Angle of the upcoming waypoints
                currVel: Current velocity of the car

            Returns:
                a : Acceleration
        """
        steepness = 2

        targetVel = self.maxVel * 2 / ( 1 + np.exp(steepness * abs(theta)))

        targetVel = max(self.minVel, min(self.maxVel, targetVel))

        a = KP * (targetVel - currVel)

        # pid = PID(KP, KI, KD, setpoint=targetVel)

        # a = pid(currVel)

        return a
    
    def pidControl(self, targetVel):

        # pid = PID(KP, KI, KD, setpoint=targetVel)

        # a = pid(self.v)

        v_error = targetVel - self.v

        self.integral += v_error * TIMERVALUE

        a = KP * v_error + KI * self.integral + KD * (v_error - self.prev_error) / TIMERVALUE

        self.prev_error = v_error

        return a

    def computeFutureTheta(self, xCheckFrontAxle):
        """ Computes the angle of the upcoming waypoints """
        theta = 0
        alpha = 0.3

        for i in range(3):
            ind = (self.ind + i) % len(self.waypoints)
            currWaypoint = self.waypoints[ind]
            waypointCarFrame = self.rotate_point([0, 0, 2 * np.pi - self.yaw], [currWaypoint[0] - xCheckFrontAxle[0], currWaypoint[1] - xCheckFrontAxle[1], 0])
            thetaTemp = abs(np.arctan2(waypointCarFrame[1], waypointCarFrame[0]))
            thetaTemp *= (1 - alpha) * (alpha ** i)
            theta += thetaTemp

        return theta

    def rotate_point(self,xCheck,cone):
        """Tranforms the coordinates of the cone to the global frame after applying rotation matrix to the cone coordinates in the car frame

        Args:
            xCheck : Car pose
            cone : Cone coordinates in the car frame
        
        Returns:
            x,y : Cone coordinates in the global frame
        """

        theta = xCheck[2]
        rotationMat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]], dtype = "float")

        newCone = rotationMat.dot(cone[:2])
        newCone = newCone.flatten()
        return newCone[0],newCone[1]

    def calculate_radius(self, a, b, c):
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        x3, y3 = c[0], c[1]
        
        a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        c = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        
        q = (a**2 + b**2 - c**2)/(2*a*b)

        # checks q value to prevent division by 0
        if abs(q) == 1:
            r = 100
        else:
            r = c/(2*np.sqrt(1-q**2))
        
        return r

    def getWaypointSign(self, xCheck, prevWaypoint, currWaypoint):
        slopeWaypoints = (currWaypoint[1] - prevWaypoint[1]) / (currWaypoint[0] - prevWaypoint[0] + 1e-6)
        perpLineSlope = -1 / slopeWaypoints
        a, b, c = perpLineSlope, -1, currWaypoint[1] - perpLineSlope * currWaypoint[0]
        temp = a * xCheck[0] + b * xCheck[1] + c
        currSign = temp / abs(temp)
        return currSign

    def checkIfWaypointCrossed(self, xCheck, prevWaypoint, currWaypoint, waypointSign):
        currSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
        if (currSign * waypointSign) < 0:
            return True
        return False

    def isSame(self,waypoint1,waypoint2):
        """Checks if two waypoints are the same by checking if the x and y values of the cones are within a threshold
        
        Args:
            waypoint1 : Waypoint 1
            waypoint2 : Waypoint 2

        Returns:
            1 if same, 0 if not
        """

        threshold = 1.5

        if (np.sqrt((waypoint1[0] - waypoint2[0])**2 + (waypoint1[1] - waypoint2[1])**2) < threshold):
            return 1
        return 0

    def getWeightedAvg(self, cone1, cone2):
        """
            Calculates the weighted average of two cones
            
            Args:
                cone1 : Cone 1
                cone2 : Cone 2
            
            Returns:
                [avgX, avgY, color, side] : Weighted average of the two waypoints
        """

        weight = 0.2
        avgX = round(((weight * cone1[0] + cone2[0])/(1+weight)),5)
        avgY = round(((weight * cone1[1] + cone2[1])/(1+weight)),5)
        return [avgX, avgY, cone1[2], cone1[3]]

    def getWeightedAvgWaypoint(self,waypoint1,waypoint2):
        """
            Calculates the weighted average of two waypoints
            
            Args:
                waypoint1 : Waypoint 1
                waypoint2 : Waypoint 2
            
            Returns:
                [avgX,avgY] : Weighted average of the two waypoints
        """

        weight = 0.2
        avgX = round(((weight * waypoint1[0] + waypoint2[0])/(1+weight)),5)
        avgY = round(((weight * waypoint1[1] + waypoint2[1])/(1+weight)),5)
        return [avgX,avgY]

    def createWaypointsMap(self, waypoints):
        """Creates the map of the waypoints.

        HAVE TO EDIT DOCUMENTATION
        """

        if waypoints == self.prevWaypoints:
            print("No new waypoints found")
            return
        
        self.prevWaypoints = waypoints

        sectors = self.waypointSectors
        
        for waypoint in waypoints:

            waypointRange = np.sqrt(waypoint[0]**2 + waypoint[1]**2)

            sectorIndex = int(waypointRange // self.sectorRangeThreshold)

            while (sectorIndex > len(sectors)-1):
                sectors.append([])
            
            sameFound = 0
            for i in range(sectorIndex - 1, sectorIndex + 2):
                if (i >= len(sectors)):
                    continue
                sect = sectors[i]
                for j in sect:
                    if (self.isSame(j, waypoint)):
                        sameFound = 1
                        break

            if sameFound == 0:
                temp_ind = len(self.waypoints) - 1
                while (len(self.waypoints) > 0 and temp_ind >= self.ind and self.euclidean_dist(self.xCheck, waypoint) < self.euclidean_dist(self.xCheck, self.waypoints[temp_ind])):
                    temp_ind -= 1

                temp_ind += 1

                self.waypoints.insert(temp_ind, waypoint)
                sect.append(waypoint)

        self.waypointSectors = sectors

        

    def compareWaypoints(self, waypoint1, waypoint2, xCheck) -> int:
        """ Compares two waypoints based on distance from car """
            
        dist1 = np.sqrt((waypoint1[0] - xCheck[0])**2 + (waypoint1[1] - xCheck[1])**2)
        dist2 = np.sqrt((waypoint2[0] - xCheck[0])**2 + (waypoint2[1] - xCheck[1])**2)

        if dist1 < dist2:
            return -1
        elif dist1 > dist2:
            return 1
        else:
            return 0
    
    def insertWaypoint(self, waypoints, waypoint, xCheck):
        """ Inserts a waypoint into the list of waypoints using binary search"""
            
        if len(waypoints) == 0:
            waypoints.append(waypoint)
            return

        low = 0
        high = len(waypoints) - 1

        while low <= high:
            mid = (low + high) // 2
            res = self.compareWaypoints(waypoints[mid], waypoint, xCheck)
            if res == 1:
                high = mid - 1
            elif res == -1:
                low = mid + 1
            else:
                return

        waypoints.insert(low, waypoint)

    def createMap(self,arrCones):
        """
            Creates the map of the cones.

            There is a condition to check if new cones are seen to prevent unnecessary computations.
            New sectors are created as needed.

            Theory behind mapping: 
            The track is divided into sectors based on the distance of the cones from the start position of car. 
            Each sector is a concentric circle, and the region between the two circles is a sector.
            The sectors are further divided into Blue and Yellow sectors based on the color of the cones.
            The distance of the cone is divided by the radius of the sector and the integer part is taken to get the sector index.

            Example: 
            Cone with distance from start position of car = 5.5m divided by sector radius 5m will give index 1. This is correct because the cones 
            in sector 0 will be the ones with distance less than radius 5m. The cones in sector 1 will be the ones with distance between 5m and 10m.

            If a similar cone is already found in the map, then a weighted average is taken and the map is updated. 

            Args: 
                arrCones : Array of cones
        """

        if arrCones == self.prevArrCones:
            return
        
        self.prevArrCones = arrCones

        sectorsBlue = self.sectorsBlue
        sectorsYellow = self.sectorsYellow

        for con in arrCones:
            cone = [con[0],con[1],con[2],con[3]]
            color = con[2]

            coneRange = np.sqrt(cone[0]**2 + cone[1]**2)

            sectorIndex = int(coneRange // self.sectorRangeThreshold)

            if (color == "Blue"):
                sectors = sectorsBlue
            else:
                sectors = sectorsYellow

            while (sectorIndex > len(sectors)-1):
                sectors.append([])
            
            sect = sectors[sectorIndex]
            sameFound = 0
            for j in range(len(sect)):
                if self.isSame(cone,sect[j]):
                    sameFound = 1
                    sect[j] = self.getWeightedAvg(cone,sect[j])
                    break
            if sameFound == 0:
                sect.append(cone)

        self.sectorsBlue = sectorsBlue
        self.sectorsYellow = sectorsYellow
        #self.saveWaypointsAndConesToCSV(arrCones)

    '''def saveWaypointsAndConesToCSV(self, cones, filename='WaypointsFromPathPlanning2.csv'):

        with open(filename, 'w', newline='') as f:
            csvwriter = csv.writer(f, delimiter=',')

            for waypoint in self.waypoints:
                csvwriter.writerow([waypoint[0], waypoint[1], 'Waypoint'])

            for cone in self.sectorsYellow:
                for c in cone:
                    csvwriter.writerow([c[0], c[1], 'Yellow Cone'])
            
            for cone in self.sectorsBlue:
                for c in cone:
                    csvwriter.writerow([c[0], c[1], 'Blue Cone'])
            
            for cone in cones:
                if cone not in [c for sector in self.sectorsYellow for c in sector] and cone not in [c for sector in self.sectorsBlue for c in sector]:
                    csvwriter.writerow([cone[0], cone[1], 'Orange Cone'])'''


    def getNeighbouringCones(self, xCheck):
        arrCones = []
        xCheckRange = np.sqrt(xCheck[0]**2 + xCheck[1]**2)

        sectorIndex = int(xCheckRange // self.sectorRangeThreshold)

        def getConesFromNearbySectors(sectorIndex, sectors):
            arrCones = []
            for i in range(sectorIndex-1, sectorIndex+2):
                if (i < len(sectors)):
                    sect = sectors[i]
                    for j in range(len(sect)):
                        # if (np.sqrt((xCheck[0] - sect[j][0])**2 + (xCheck[1] - sect[j][1])**2) > 8):
                        #     continue
                        insert = True
                        for k in range(len(arrCones)):
                            if self.isSame(sect[j], arrCones[k]):
                                insert = False
                                break
                        if insert:
                            arrCones.append(sect[j])
            return arrCones

        sectors = self.sectorsYellow
        arrCones = getConesFromNearbySectors(sectorIndex, sectors)

        sectors = self.sectorsBlue
        arrCones += getConesFromNearbySectors(sectorIndex, sectors)

        return arrCones

    def cte_function(self, x):
        xCheck = [self.xCheck[0] + WHEELBASE * np.cos(self.yaw), self.xCheck[1] + WHEELBASE * np.sin(self.yaw), self.xCheck[2]]
        cost = 0
        steerSign = np.sign(x[1])
        waypointSign = self.waypointSign
        DT = 0.1
        ind = self.ind
        currWaypoint = self.waypoints[ind]
        prevWaypoint = self.waypoints[ind - 1]
        alpha = 0.3
        carSteer = self.carSteer

        for i in range(5):
            carSteer += min(DT * np.pi, abs(x[1])) * steerSign
            xCheck[2] += carSteer
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])

            if (self.checkIfWaypointCrossed(xCheck, prevWaypoint, currWaypoint, waypointSign)):
                ind = (ind + 1) % len(self.waypoints)
                currWaypoint = self.waypoints[ind]
                prevWaypoint = self.waypoints[ind - 1]
                waypointSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
            
            slope = (currWaypoint[1] - prevWaypoint[1]) / (currWaypoint[0] - prevWaypoint[0] + 1e-6)

            # Equation of line joining the two waypoints
            a, b, c = slope, -1, prevWaypoint[1] - slope * prevWaypoint[0]
            perp_dist = abs(a * xCheck[0] + b * xCheck[1] + c) / np.sqrt(a**2 + b**2)

            cost += (1 - alpha) * alpha**i * perp_dist

            #cost += (1 - alpha) * alpha**i * abs((prevWaypoint[1] - currWaypoint[1]) * xCheck[0] + (currWaypoint[0] - prevWaypoint[0]) * xCheck[1] * DT + (prevWaypoint[0] * currWaypoint[1] - currWaypoint[0] * prevWaypoint[1])) / np.sqrt((prevWaypoint[1] - currWaypoint[1])**2 + (currWaypoint[0] - prevWaypoint[0])**2)
        
        return cost

    def delta_steer_function(self, x):
        return abs(x[1] - self.carSteer)

    def theta_next_function(self, x):
        xCheck = [self.xCheck[0] + WHEELBASE * np.cos(self.yaw), self.xCheck[1] + WHEELBASE * np.sin(self.yaw), self.xCheck[2]]
        cost = 0
        steerSign = np.sign(x[1])
        waypointSign = self.waypointSign
        DT = 0.2
        ind = self.ind
        nextWaypoint = self.waypoints[(ind + 1) % len(self.waypoints)]
        alpha = 0.3
        carSteer = self.carSteer

        waypointCarFrame = self.rotate_point([0, 0, 2 * np.pi - self.yaw], [nextWaypoint[0] - xCheck[0], nextWaypoint[1] - xCheck[1], 0])
        thetaTemp = abs(np.arctan2(waypointCarFrame[1], waypointCarFrame[0]))

        cost += thetaTemp

        return cost

    def dist_from_waypoint_function(self, x):
        xCheck = [self.xCheck[0] + WHEELBASE * np.cos(self.yaw), self.xCheck[1] + WHEELBASE * np.sin(self.yaw), self.xCheck[2]]
        cost = 0
        steerSign = np.sign(x[1])
        waypointSign = self.waypointSign
        DT = 0.1
        ind = self.ind
        currWaypoint = self.waypoints[ind]
        prevWaypoint = self.waypoints[ind - 1]
        alpha = 0.5
        carSteer = self.carSteer

        for i in range(10):
            carSteer += min(DT * np.pi, abs(x[1])) * steerSign
            xCheck[2] += carSteer
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])

            if (self.checkIfWaypointCrossed(xCheck, prevWaypoint, currWaypoint, waypointSign)):
                ind = (ind + 1) % len(self.waypoints)
                currWaypoint = self.waypoints[ind]
                prevWaypoint = self.waypoints[ind - 1]
                waypointSign = self.getWaypointSign(xCheck, prevWaypoint, currWaypoint)
            
            cost += (1 - alpha) * alpha**i * self.euclidean_dist(xCheck, currWaypoint)
        
        return cost

    def dist_function(self, x, cones):
        """
            Calculates the cost based on the distance from the both edges of the track
        """
        cost = 0
        xCheck = [self.xCheck[0] + WHEELBASE * np.cos(self.yaw), self.xCheck[1] + WHEELBASE * np.sin(self.yaw), self.xCheck[2]]
        DT = 0.2
        steerSign = np.sign(x[1])
        carSteer = 0
        alpha = 0.3

        blueCones = []
        yellowCones = []

        for cone in cones:
            if cone[2] == "Blue":
                blueCones.append(cone[0:2])
            elif cone[2] == "Yellow":
                yellowCones.append(cone[0:2])

        blueCones = np.array(blueCones)
        yellowCones = np.array(yellowCones)

        def distFromEnd(color):
            # Obtain two closest blue cones to the car
            if (color == "Blue"):
                conesCopy = blueCones
            else:
                conesCopy = yellowCones

            if (len(conesCopy) < 2):
                raise Exception("Not enough cones for color:", color)

            closestCones = [(None, float('inf')), (None, float('inf'))]

            for cone in conesCopy:
                dist = np.sqrt((cone[0] - xCheck[0])**2 + (cone[1] - xCheck[1])**2)
                if dist < closestCones[0][1]:
                    closestCones = [(cone, dist), closestCones[0]]
                elif dist < closestCones[1][1]:
                    closestCones[1] = (cone, dist)

            # Find perpendicular distance from the car to the line joining the two closest blue cones

            x1, y1 = closestCones[0][0]
            x2, y2 = closestCones[1][0]
            numerator = abs((y2 - y1) * xCheck[0] - (x2 - x1) * xCheck[1] + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            dist = numerator / denominator
            return dist

        for i in range(3):

            carSteer += min(DT * np.pi, abs(x[1])) * steerSign
            xCheck[2] += carSteer
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])

            try:

                dist_blue = distFromEnd("Blue")
                dist_yellow = distFromEnd("Yellow")

            except Exception as e:
                raise Exception(f"Error in dist from edges of track function. {e} Neighbouring cones: ", cones)

            cost += abs(dist_blue - dist_yellow)  * (1 - alpha) * (alpha ** i)

        return cost

    def diff_from_max_travel(self, x):
        distTravelled, maxTravellableDist = 0, 0
        xCheck = [self.xCheck[0] + WHEELBASE * np.cos(self.yaw), self.xCheck[1] + WHEELBASE * np.sin(self.yaw), self.xCheck[2]]
        DT = 0.2
        oldXCheck = deepcopy(xCheck)
        steerSign = np.sign(x[1])
        carSteer = 0
        alpha = 0.3

        for i in range(3):

            carSteer += min(DT * np.pi, abs(x[1])) * steerSign
            xCheck[2] += carSteer
            xCheck[0] += DT * x[0] * np.cos(xCheck[2])
            xCheck[1] += DT * x[0] * np.sin(xCheck[2])

            maxTravellableDist += DT * self.maxVel
            distTravelled += self.euclidean_dist(xCheck, oldXCheck)
            oldXCheck = xCheck

        return abs(distTravelled - maxTravellableDist)
    


    def mpc_cost_function(self, x):
        return 4 * self.cte_function(x) + 10 * self.dist_from_waypoint_function(x) + 1.5 * self.delta_steer_function(x)  + 250 * self.theta_next_function(x)

    def mpc_cost_function_alt(self, x, cones):
        return  8 * self.dist_function(x, cones) + 5 * self.delta_steer_function(x) + 1 * self.diff_from_max_travel(x)
    
    def start_path_optimization(self):
        """Start the path optimization in a separate thread"""

        if self.optimization_in_progress:
            return
        
        def optimize_wrapper():
            try:
                waypoints_copy = self.waypoints.copy()

                optimized_waypoints = self.optimizer.optimize_path(
                    waypoints_copy,
                )

                if optimized_waypoints is not None:
                    with PathOptimizer._optimization_lock:
                        self.waypoints = optimized_waypoints
                        PathOptimizer.optimization_complete = True
            except Exception as e:
                print("Error in path optimization: ", e)
            finally:
                self.optimization_requested = False
        
        self.optimization_in_progress = True # Set flag to indicate optimization in progress and request optimization only once

        PathOptimizer.optimization_thread = threading.Thread(target=optimize_wrapper)
        PathOptimizer.optimization_thread.daemon = True
        PathOptimizer.optimization_thread.start()

    def update_optimized_path(self):       
        
        if not hasattr(self, 'optimized_waypoints') or not self.optimization_complete:
            print("Optimization incomplete or no optimized waypoints")
            return
        
        with PathOptimizer._optimization_lock:
            try:
                if not hasattr(self, 'optimized_waypoints') or self.optimized_waypoints is None:
                    print("Optimized waypoints not available after lock")
                    return

                curr_pos = [self.xCheck[0], self.xCheck[1]]

                self.waypoints = []
                self.waypointSectors = [[]]
                            
                for waypoint in self.optimized_waypoints:
                    self.createWaypointsMap([waypoint])

                min_dist = float('inf')
                new_ind = 0
                
                for i, waypoint in enumerate(self.waypoints):
                    dist = self.euclidean_dist(curr_pos, waypoint)
                    # Only consider waypoints ahead of the car
                    waypoint_car_frame = self.rotate_point(
                        [0, 0, 2 * np.pi - self.yaw],
                        [waypoint[0] - curr_pos[0], waypoint[1] - curr_pos[1]]
                    )
                    if waypoint_car_frame[0] > 0 and dist < min_dist:
                        min_dist = dist
                        new_ind = i

                # Update indices and flags
                self.ind = new_ind
                self.ok = True
                self.started = True
                self.waypointSign = self.getWaypointSign(
                    [curr_pos[0], curr_pos[1], self.yaw],
                    self.waypoints[self.ind - 1],
                    self.waypoints[self.ind]
                )

                # Reset optimization flags
                self.optimization_complete = False
                self.optimized_waypoints = None

            except Exception as e:
                print("Error in updating optimized path: ", e)
                
            finally:
                print("Optimization update complete")



    def mainLoop(self):

        xCheck = self.xCheck

        # PATH PLANNING

        lk = self.lk
        
        if (len(lk) > 0) and self.laps < 2:
            arrCones = []
            for cone in lk:
                if (np.sqrt(cone[0]**2 + cone[1]**2) < 9):
                    coneRotationFrameUpdated = self.rotate_point(xCheck,cone)
                    coneRotationFrameUpdated = [coneRotationFrameUpdated[0] + xCheck[0], coneRotationFrameUpdated[1] + xCheck[1], cone[2], cone[3]]
                    arrCones.append(coneRotationFrameUpdated)
            self.createMap(arrCones)

            cones = self.getNeighbouringCones(xCheck)
            # cones = arrCones

            try:
                waypoints = []
                conesForDelaunay = []
                for i in cones:
                    i = [i[0] - xCheck[0], i[1] - xCheck[1]]
                    coneUpdatedRotationFrame = self.rotate_point([0, 0, 2 * np.pi - self.yaw], i)
                    if (coneRotationFrameUpdated[0] < -1):
                        continue
                    conesForDelaunay.append([i[0], i[1]])

                dt_tris = Delaunay(conesForDelaunay)

                for i in dt_tris.simplices:
                    x = np.array(i)
                    cone1 = cones[x[0]]
                    cone2 = cones[x[1]]
                    cone3 = cones[x[2]]
                    if (cone1[2] == cone2[2] == "Big Orange"):
                        waypoint = [((cone1[0] + cone2[0]) / 2) , ((cone1[1] + cone2[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                    if (cone2[2] == cone3[2] == "Big Orange"):
                        waypoint = [((cone2[0] + cone3[0]) / 2) , ((cone2[1] + cone3[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                    if (cone1[2] == cone3[2] == "Big Orange"):
                        waypoint = [((cone1[0] + cone3[0]) / 2) , ((cone1[1] + cone3[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                    if (cone1[2] != cone2[2] and (cone1[2] != "Big Orange" and cone2[2] != "Big Orange") and (np.sqrt((cone1[0] - cone2[0])**2 + (cone1[1] - cone2[1])**2) < 7)) :
                        waypoint = [((cone1[0] + cone2[0]) / 2) , ((cone1[1] + cone2[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                    if (cone2[2] != cone3[2] and (cone2[2] != "Big Orange" and cone3[2] != "Big Orange") and np.sqrt((cone2[0] - cone3[0])**2 + (cone2[1] - cone3[1])**2) < 7) :  
                        waypoint = [((cone2[0] + cone3[0]) / 2) , ((cone2[1] + cone3[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                    if (cone1[2] != cone3[2] and (cone1[2] != "Big Orange" and cone3[2] != "Big Orange") and np.sqrt((cone1[0] - cone3[0])**2 + (cone1[1] - cone3[1])**2) < 7) :
                        waypoint = [((cone1[0] + cone3[0]) / 2) , ((cone1[1] + cone3[1]) / 2)]
                        self.insertWaypoint(waypoints, waypoint, xCheck)
                self.createWaypointsMap(waypoints)
            except Exception as e:
                print("Error in Delaunay Triangulation: ", e)

        # CONTROLS
        cones = self.getNeighbouringCones(xCheck)
        
        xCheckFrontAxle = [xCheck[0] + WHEELBASE * np.cos(self.yaw), xCheck[1] + WHEELBASE * np.sin(self.yaw), xCheck[2]]
        accel = 0
        steerAngle = 0
        try:

            if not self.ok:
                if self.ind != len(self.waypoints) - 1:
                    for i in range(self.ind + 1, len(self.waypoints)):
                        waypoint_updated_frame = self.rotate_point(
                            [0, 0, 2 * np.pi - self.yaw],
                            [self.waypoints[i][0] - xCheck[0], 
                            self.waypoints[i][1] - xCheck[1]]
                        )
                        if waypoint_updated_frame[0] > 0:
                            self.ind = i
                            self.ok = True
                            self.started = True
                            break

            if self.ok and self.checkIfWaypointCrossed(
                xCheckFrontAxle, 
                self.waypoints[self.ind - 1], 
                self.waypoints[self.ind], 
                self.waypointSign
            ):
                next_ind = (self.ind + 1) % len(self.waypoints)
                waypoint_updated_frame = self.rotate_point(
                    [0, 0, 2 * np.pi - self.yaw],
                    [self.waypoints[next_ind][0] - xCheck[0], 
                    self.waypoints[next_ind][1] - xCheck[1]]
                )

                # Modified lap transition logic
                if next_ind == 0:
                    if waypoint_updated_frame[0] > 0 and self.started:
                        self.ind = next_ind
                        self.ok = True
                        self.lapEndClose = True
                    else:
                        self.ok = False
                else:
                    self.ind = next_ind
                    self.ok = True
                    self.started = True

            # Modified lap completion logic
            if self.lapEndClose and self.ind == 1:
                timeEnd = time.time()
                timeTakenInSeconds = timeEnd - self.timeStart
                self.laps += 1
                print(f"TIME TAKEN: {timeEnd - self.timeStart}")
                print(f"AVERAGE VELOCITY: {DISTOFSMALLTRACK / timeTakenInSeconds}")
                self.maxVel = 6.5
                
                self.timeStart = time.time()
                self.started = False
                self.lapEndClose = False
                self.ok = True

            if self.laps == 2 and self.optimizer_status == True:
                
                print("Starting path optimization...")
                self.optimizer_status = False
                self.start_path_optimization()
                

            if hasattr(self, 'optimization_complete') and self.optimization_complete:
                print("Optimization completed, Updating ...")
                self.update_optimized_path()
            
            

            if (self.ok):
                self.waypointSign = self.getWaypointSign(xCheckFrontAxle, self.waypoints[self.ind - 1], self.waypoints[self.ind])

            currWaypoint = self.waypoints[self.ind]
            
            prevWaypoint = self.waypoints[self.ind - 1]

            initial_guess = np.array([3, 0])
            bounds = Bounds([3, -0.52], [self.maxVel, 0.52])

            if self.previous_result is None:
               self.previous_result = np.array([3, 0])

            if self.ok:
                #print("USING NORMAL FUNCTION")
                initial_guess = self.previous_result
                result = minimize(self.mpc_cost_function, initial_guess, method='SLSQP', bounds=bounds)
                self.previous_result = np.array(result.x) if result.success else self.previous_result
                
            
            else:
                #print("USING ALTERNATE FUNCTION")
                bounds = Bounds([1, -0.52], [2, 0.52])
                initial_guess = np.array([1.5, 0])
                result = minimize(self.mpc_cost_function_alt, initial_guess, args=(cones,), method='SLSQP', bounds=bounds)

            if (result.success == False):
                raise Exception("Optimization failed")
            
            
            desiredVelocity = result.x[0]
            steerAngle = result.x[1]

            if steerAngle > MAXSTEER:
                steerAngle = MAXSTEER
            elif steerAngle < - MAXSTEER:
                steerAngle = - MAXSTEER
                    
            
            #print("Desired Velocity: ", desiredVelocity, "Steer Angle: ", steerAngle)

            accel = self.pidControl(desiredVelocity)
            data = AckermannDriveStamped()
            data.drive.steering_angle = float(steerAngle)
            data.drive.acceleration = float(accel)
            self.controlsPub.publish(data)

        except Exception as e:
           # print("Error in controls ", e)
            accel = 0.1
            data = AckermannDriveStamped()
            data.drive.steering_angle = float(0)
            data.drive.acceleration = float(accel)
            self.controlsPub.publish(data)

        vehicle_state = VehicleState()
        vehicle_state.x = float(xCheck[0])
        vehicle_state.y = float(xCheck[1])
        vehicle_state.yaw = float(xCheck[2])
        vehicle_state.velocity = float(self.v)
        vehicle_state.steer_angle = float(self.carSteer)

        control_command = ControlCommand()
        control_command.acceleration = float(accel)
        control_command.steer_angle = float(steerAngle)

        cones_seen = ConesList()

        for cone in cones:
            cone_msg = Cone()
            cone_msg.x = float(cone[0])
            cone_msg.y = float(cone[1])
            cone_msg.type = str(cone[2])
            cones_seen.cones.append(cone_msg)

        map_msg = PointsList()

        for sector in self.sectorsBlue:
            for cone in sector:
                point = Point()
                point.x = float(cone[0])
                point.y = float(cone[1])
                map_msg.points.append(point)
        
        for sector in self.sectorsYellow:
            for cone in sector:
                point = Point()
                point.x = float(cone[0])
                point.y = float(cone[1])
                map_msg.points.append(point)
            
        waypoints_msg = PointsList()

        for waypoint in self.waypoints:
            point = Point()
            point.x = float(waypoint[0])
            point.y = float(waypoint[1])
            waypoints_msg.points.append(point)
        
        monitor_data_msg = MonitorData()
        monitor_data_msg.vehicle_state = vehicle_state
        monitor_data_msg.control_command = control_command
        monitor_data_msg.cones_seen = cones_seen
        monitor_data_msg.map = map_msg
        monitor_data_msg.waypoints_list = waypoints_msg

        self.monitor_pub.publish(monitor_data_msg)
            
    def loopSetter(self):
        """
        This function is called when the node is started. It runs the main loop at a fixed rate.
        """

        timerPeriod = 0.02

        try:

            self.timer = self.create_timer(timerPeriod, self.mainLoop)

        except KeyboardInterrupt:
            print("ROS Interrupt Exception")
            exit(1)

class PathOptimizer:

    _optimization_lock = threading.Lock()
    def __init__(self):
        self.smoothing_factor = 0.7
        self.min_distance = 0.15
        self.car_width = 1.58  
        self.safety_margin = 0.2
        self.optimization_complete = False
        self.optimization_thread = None

    def optimize_path(self, waypoints):
        """Optimize path for minimum curvature with clearance constraints"""
        try:
            if waypoints is None or len(waypoints) < 2:
                print("Error: Not enough waypoints to optimize path.")
                return None

            points = np.array([(wp[0], wp[1]) for wp in waypoints])           

            tck, u = splprep([points[:,0], points[:,1]], s=self.smoothing_factor, per=True)

            u_new = np.linspace(0, 1, len(waypoints) * 2)
            x_spline, y_spline = splev(u_new, tck)

            dx = np.gradient(x_spline)
            dy = np.gradient(y_spline)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5

            weights = 1 / (1 + curvature)

            tck, u = splprep([x_spline, y_spline],
                           w=weights, 
                           s=self.smoothing_factor*len(weights), 
                           per=True)

            u_final = np.linspace(0, 1, len(waypoints))
            x_final, y_final = splev(u_final, tck)
            
            optimized_path = np.column_stack((x_final, y_final))
            
            self.optimization_complete = True
            
            return optimized_path

        except Exception as e:
            print(f"Error in optimize_path: {str(e)}")
            return None
    
def main(args=None):
    rclpy.init(args=args)
    obj = Controls()
    rclpy.spin(obj)
    obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
