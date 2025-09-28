from scipy.optimize import minimize

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
from coneperception import ConeDetection


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
    for row in csv_reader:
        cones.append([(row[0]), float(row[1]),float(row[2])])

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


"""
Mass = 225 kg
g = 9.81
Inertial Force I_z = 31.27
kinematics:
  l: 1.58           # Wheelbase              [m]
  b_F: 0.869        # From COG to front axle [m]
  b_R: 0.711        # From COG to rear axle  [m]
  w_front: 0.45     # Percentage of weight front
  axle_width: 1.4
tire:
  tire_coefficient: 1.0
  B: 12.56
  C: -1.38
  D: 1.60
  E: -0.58
  radius: 0.2525
aero:
  C_Down: 1.9 # F_Downforce = C_Downforce*v_x^2
  C_drag: 1   # F_Drag = C_Drag*v_x^2
"""

LR = 0.71
WHEELBASE = 1.58
DISTOFSMALLTRACK = 101.1 # Distance of small track in meters
MAX_STEER = 0.52 # Maximum steering angle
MAX_ACCELERATION = 1
MIN_ACCELERATION = -1
TIMER_PERIOD = 1/40 

KP = 2 # Constant for proportional control
KI = 4
KD = 1
KSTEER = 0.2 # Constant for steering control
KSOFTENING = 1 # Constant for softening control
MAXSTEER = 0.52 # Maximum steering angle
DISTOFSMALLTRACK = 101.1 # Distance of small track in meters
TIMERVALUE = 0.02



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

class MPC(Communication):
    def __init__(self):
        print("MPC constructor called")
        super().__init__()
        self.integral = 0
        self.prev_error = 0

    
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
    
    def euclidean_dist(self, point1, point2):
        """From your working code"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def rotate_point(self, xCheck, cone):
        """EXACT from your working code"""
        theta = xCheck[2]
        rotationMat = np.array([[np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]], dtype="float")
        newCone = rotationMat.dot(cone[:2])
        newCone = newCone.flatten()
        return newCone[0], newCone[1]

    def updateCones(self):
    #Updates the cone locations in the car frame as seen by the camera
        lk = []
        
        with open(os.path.join(os.path.dirname(__file__), '..', 'cones.csv'), 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header

            for row in csv_reader:
                tag = row[0].lower().strip()
                x = float(row[1])
                y = float(row[2])

                if tag == "blue":
                    lk.append([x, y, "Blue", "Left"])
                elif tag == "yellow":
                    lk.append([x, y, "Yellow", "Right"])
                elif tag in ["orange", "big_orange"]:
                    side = "Right" if y < 0 else "Left"
                    lk.append([x, y, "Big Orange", side])
                else:
                    continue

        if self.lk == getattr(self, "lk", []):
            return

        self.prevlk = getattr(self, "lk", [])
        self.lk = lk


    

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
    
    def createWaypointsMap(self):
        """Creates the map of the waypoints.

        HAVE TO EDIT DOCUMENTATION
        """

        if waypointlist == self.prevWaypoints:
            print("No new waypoints found")
            return
        
        self.prevWaypoints = waypointlist

        sectors = self.waypointSectors
        
        for waypoint in waypointlist:

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


class Mpc(Communication):

    def __init__(self):
        print("mpc constructor is caleld")
        super().__init__()
        self.old_steer, self.old_acceleration = 0, 0

        #Cache of sorts - for finding x,y for every prediction by scipy
        self.prevX, self.prevY, self.prevA, self.prevDelta = 0,0,0,0 

        self.communication = Communication()
        self.waypoints = Waypoints()
        self.localisation = Localisation()
        self.perception = Perception()
        self.localisation.CreateMap(cones)

        self.t = 0.1
        self.maxIter = 40        

    def dynamic_slip(self,vX, vY,steer):
        return 1.0 * (np.arctan(0.46 * np.tan(steer)) - np.arctan2(vY,vX))**2

    def SlipFunction(self, carSteer):
        qBeta = 2
        lr = lf =  LR

        dynamic_slip = np.arctan2(self.v_y,self.v_x)
        kinematic_slip = np.arctan2(np.tan(carSteer) * lr , (lr + lf))

        slipCost = qBeta * ((kinematic_slip - dynamic_slip)**2)
        return slipCost

    """def ObjectiveFunction(self, params):
        a, steer = params
        
        carSteer = self.old_steer
    
        #Right now current as it will get updated every iteration (including first iteration)
        predictedYaw = self.yaw
        currentX, currentY = self.x, self.y
        currentVelocity = self.v
        currentVelocityX, currentVelocityY = self.v_x, self.v_y

        costfn = 0 
        weights = [0.75,0.15,0.05]
        
        
        Steer is the actual angle of the steering wheel and NOT the extra angle to turn (deltaSteer)
        

        #carSteer = steer #ONLY FOR NOW -> sim doesnt give steer

        #current -> current values => values of the prev iteration
        #predicted -> predicted values of that iteration
        
        #weights = [1]
        for i in range(1):
            outOfBounds = False
            deltaSteer = steer - carSteer
            
            carSteer += min(self.t*1.04, abs(deltaSteer))*np.sign(deltaSteer)
            
            #carSteer += deltaSteer

            #beta = np.arctan(LR*np.tan(carSteer)/WHEELBASE)
            # Combined slip angle calculation considering both lateral velocity and steering angle
            #beta = np.arctan2(currentVelocityY + LR * np.tan(carSteer), currentVelocityX)
            beta = np.arctan(LR * np.tan(carSteer))
            


            yawRate = currentVelocity * np.cos(beta) * np.tan(carSteer)/WHEELBASE
            predictedYaw += yawRate #* self.t
            

            #yaw -> Predicted yaw
            #predictedYaw += carSteer

            a_x = a * np.cos(predictedYaw+beta)
            a_y = a * np.sin(predictedYaw+beta)


            predictedX = currentX + currentVelocityX*self.t + 0.5*a_x*self.t**2
            predictedY = currentY + currentVelocityY*self.t + 0.5*a_y*self.t**2
            

            predictedVelocity = currentVelocity + a*self.t
            referenceVelocity = self.waypoints.GetRefVel([predictedX,predictedY], 1)
            error = referenceVelocity - np.abs(predictedVelocity)
            
            #print("Vref",referenceVelocity)
            #predictedAvgVelError =  (ReferenceVelocity + PredictedVelocity)/(self.waypoints.v_count+1) - (self.waypoints.v_sum + PredictedVelocity)/(self.waypoints.v_count+1)

            distNotTravelled = referenceVelocity * self.t - ((predictedX-currentX)**2 + (predictedY-currentY)**2)**0.5
           
            crosstrack_error, heading_error = self.waypoints.GetError(predictedX, predictedY, predictedYaw)     #crosstrack error is not included            
            

            # predictedDistFromLeft, predictedDistFromRight = self.localisation.DistanceFromSides(predictedX,predictedY)
            
            #kl, kr = self.localisation.get_spline_and_curvature(x,y)

            #slipCost = self.SlipFunction(carSteer)

            slipCost = self.dynamic_slip(currentVelocityX, currentVelocityY, carSteer)
            

            #Lateral Control
            tempCostFunction = 17*np.abs(heading_error) + np.abs(crosstrack_error) # + 8*((np.abs(predictedDistFromLeft)) + (np.abs(predictedDistFromRight))) #+ 0.5*np.abs(deltaSteer)+ 15* (kl/(kl+kr))*(np.abs(PredictedDistFromLeft)) +15* (kr/(kl+kr))*(np.abs(PredictedDistFromRight))  # #+ 
            
            #Longitudinal Control
            tempCostFunction += 3*np.abs(distNotTravelled) + 4*np.abs(error) + 2*np.abs(slipCost)#+ 4*np.abs(predictedAvgVelError)
            

            #Failsafes cause constraints don't work
            # if predictedVelocity < 0.8 or currentVelocity < 0.5:
            #     tempCostFunction += 25

            # if predictedDistFromLeft*predictedDistFromRight > 0:
            #     tempCostFunction += tempCostFunction*2
            #     outOfBounds = True
            # print("checkpt in obj 13")

            # if (np.abs(predictedDistFromLeft) < 0.75 or np.abs(predictedDistFromRight) < 0.75) and outOfBounds == False:
            #     tempCostFunction += 100
            # print("checkpt in obj 14")

            currentVelocityX, currentVelocityY = predictedVelocity*np.cos(predictedYaw+beta), predictedVelocity*np.sin(predictedYaw+beta) 
            currentVelocity = predictedVelocity
            currentX, currentY = predictedX, predictedY
            costfn += tempCostFunction*weights[i]
            

        return costfn"""
    
    def ObjectiveFunction(self, params):
        a, delta = params
        
        u_x, u_y = self.v_x, self.v_y
        #𝜃_dot = self.v / R = v *tan(𝛿) *cos(𝛽) / L
        
        yaw = self.yaw
        currentX, currentY = self.x, self.y
        current_velocity = self.v

        costfn = 0 
        weights = [0.9,0.1]

        for i in range(2):

            delta+=min(self.t*1, abs(delta))* np.sign(delta)
            yaw += delta

            a_x = a * np.cos(yaw)
            a_y = a * np.sin(yaw)

            x = currentX + u_x*self.t + 0.5*a_x*self.t**2
            y = currentY + u_y*self.t + 0.5*a_y*self.t**2

            PredictedVelocity = current_velocity + a*self.t
            ReferenceVelocity = self.waypoints.GetRefVel([x,y], 1)
            error = ReferenceVelocity - np.abs(PredictedVelocity)
            predictedAvgVelError =  (ReferenceVelocity + PredictedVelocity)/(self.waypoints.v_count+1) - (self.waypoints.v_sum + PredictedVelocity)/(self.waypoints.v_count+1)

            distNotTravelled = ReferenceVelocity * self.t - ((x-currentX)**2 + (y-currentY)**2)**0.5

            #delta_steering = delta - self.currentSteer #change
            
            crosstrack_error, heading_error = self.waypoints.GetError(x, y, yaw)     #crosstrack error is not included            

            #PredictedDistFromLeft, PredictedDistFromRight = self.localisation.DistanceFromSides(x,y)

            #kl, kr = self.localisation.get_spline_and_curvature(x,y)
            
            #Lateral Control
            tempCostfn = 7*np.abs(heading_error) +  np.abs(crosstrack_error)#+ 0.9*np.abs(delta_steering) + 15* (kl/(kr+kl))*(np.abs(PredictedDistFromLeft)) +15* (kr/(kr+kl))*(np.abs(PredictedDistFromRight)) 
            
            #Longitudinal Control
            tempCostfn += 16*np.abs(distNotTravelled) #+ 5*np.abs(error) + 3*np.abs(predictedAvgVelError)

            #Failsafes cause constraints don't work
            if PredictedVelocity <50e-2 or current_velocity < 40e-2:
                tempCostfn += 30

            #if np.abs(PredictedDistFromLeft) < 0.9 or np.abs(PredictedDistFromRight) < 0.9:
                #tempCostfn += 40
            
            '''if PredictedVelocity > ReferenceVelocity:
                tempCostfn += 80

            if PredictedDistFromLeft*PredictedDistFromRight > 0:
                tempCostfn += 300'''
           

            u_x, u_y = PredictedVelocity*np.cos(yaw), PredictedVelocity*np.sin(yaw)
            current_velocity = PredictedVelocity
            currentX, currentY = x,y
            costfn += tempCostfn*weights[i]

        return costfn

    
    
    def GetPredictedPosn(self, a, delta):
        if a == self.prevA and delta == self.prevDelta:
            return self.prevX, self.prevY

        yaw = self.yaw + delta
        a_x = a * np.cos(yaw)
        a_y = a* np.sin(yaw)
        x = self.x + self.v_x*self.t +0.5*a_x*self.t**2
        y = self.y + self.v_y*self.t +0.5*a_y*self.t**2
    
        self.prevA, self.prevDelta = a, delta
        self.prevX, self.prevY = x,y
        return x,y

    def Normalize(self, angle):       
        return (angle+np.pi)%(2*np.pi)-np.pi
    
    def Calculate(self):

        if self.old_acceleration == 0:
            self.old_acceleration = MAX_ACCELERATION
        initial_guess = [self.old_acceleration,self.old_steer]

        self.bounds = [(-1,1), (-0.52, 0.52)]
        result = minimize(self.ObjectiveFunction, initial_guess, bounds=self.bounds, method = 'SLSQP')#, constraints=self.constraints)#, options={'maxiteration': 30})
        
        optimal_values = result.x

        print('cf = ', result.fun) #yipeee
        print("Number of iterations:", result.nit)
        # print(optimal_values)
        # print(result.message)

        acceleration = optimal_values[0]
        
        steer = optimal_values[1]
        #print("steering : ",steer)
        print(f"Inputted acceleration : {acceleration}, steering : {steer}")

        self.old_steer = steer
        self.old_acceleration = acceleration      

        return acceleration, steer

    def MainLoop(self):
        self.state_update()
        self.waypoints.IfCrossedWaypoint(self.x,self.y)
        
        
        acceleration, steering_angle = self.Calculate()
    
        self.communication.inputingValue(acceleration,steering_angle)        
        
                    
        append_values_to_csv([self.x, self.y])
        

        
# if self.v > 1:
        #     self.resetSim(0)

        # if self.show_monitor:
        #     self.update_monitor(acceleration, steering_angle, self.localisation, self.waypoints)
        
        
timerPeriod = 1/30
        
def main(args=None):
    # rclpy.init(args=args)
    obj = Mpc()
    #start = time.time()
    next_call = time.time()
    try:
        while True:
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
