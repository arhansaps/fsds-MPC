from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial import ConvexHull

import pandas as pd
import numpy as np
import csv

import time
import matplotlib.pyplot as plt
import os.path

from PathPlanning import Waypoints
from SLAM import Localisation

import os.path

cones = []
with open(os.path.join(os.path.dirname(__file__),'cones.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        cones.append([float(row[0]),float(row[1]),row[2],row[3]])

#Clearing the car pos file
with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'w', newline='') as file:
    pass
file.close()

def append_values_to_csv(row):
    with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'a', newline='') as file:
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
MAX_VELOCITY = 5
TIMER_PERIOD = 1/40

class Mpc(Ros):

    def __init__(self):
        print("mpc constructor is caleld")
        self.old_steer, self.old_acceleration = 0, 0

        #Cache of sorts - for finding x,y for every prediction by scipy
        self.prevX, self.prevY, self.prevA, self.prevDelta = 0,0,0,0 

        self.t = 0.1
        self.maxIter = 40

        #objects of other classes : 
        self.waypoints = Waypoints()
        self.localisation = Localisation()

        #map init :
        self.localisation.CreateMap(cones)

        #ros init:
        super().__init__()
        self.LoopSetter()

    def dynamic_slip(self,vX, vY,steer):
        return 1.0 * (np.arctan(0.46 * np.tan(steer)) - np.arctan2(vY,vX))**2

    def SlipFunction(self, carSteer):
        qBeta = 2
        lr = lf =  LR

        dynamic_slip = np.arctan2(self.v_y,self.v_x)
        kinematic_slip = np.arctan2(np.tan(carSteer) * lr , (lr + lf))

        slipCost = qBeta * ((kinematic_slip - dynamic_slip)**2)
        return slipCost

    def ObjectiveFunction(self, params):
        a, steer = params
        
        carSteer = self.currentSteer
    
        #Right now current as it will get updated every iteration (including first iteration)
        predictedYaw = self.yaw
        currentX, currentY = self.x, self.y
        currentVelocity = self.v
        currentVelocityX, currentVelocityY = self.v_x, self.v_y

        costfn = 0 
        weights = [0.75,0.15,0.05]

        """
        Steer is the actual angle of the steering wheel and NOT the extra angle to turn (deltaSteer)
        """

        #current -> current values => values of the prev iteration
        #predicted -> predicted values of that iteration

        #weights = [1]
        for i in range(3):
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
            referenceVelocity = self.waypoints.GetRefVel([predictedX,predictedY], 2)
            error = referenceVelocity - np.abs(predictedVelocity)
            #predictedAvgVelError =  (ReferenceVelocity + PredictedVelocity)/(self.waypoints.v_count+1) - (self.waypoints.v_sum + PredictedVelocity)/(self.waypoints.v_count+1)

            distNotTravelled = referenceVelocity * self.t - ((predictedX-currentX)**2 + (predictedY-currentY)**2)**0.5
           
            crosstrack_error, heading_error = self.waypoints.GetError(predictedX, predictedY, predictedYaw)     #crosstrack error is not included            

            predictedDistFromLeft, predictedDistFromRight = self.localisation.DistanceFromSides(predictedX,predictedY)
            
            #kl, kr = self.localisation.get_spline_and_curvature(x,y)

            #slipCost = self.SlipFunction(carSteer)

            slipCost = self.dynamic_slip(currentVelocityX, currentVelocityY, carSteer)

            #Lateral Control
            tempCostFunction = 17*np.abs(heading_error) + np.abs(crosstrack_error) + 0.5*np.abs(deltaSteer) + 8*((np.abs(predictedDistFromLeft)) + (np.abs(predictedDistFromRight))) #+ 15* (kl/(kl+kr))*(np.abs(PredictedDistFromLeft)) +15* (kr/(kl+kr))*(np.abs(PredictedDistFromRight))  # #+ 
            
            #Longitudinal Control
            tempCostFunction += 3*np.abs(distNotTravelled) + 4*np.abs(error) + 2*np.abs(slipCost)#+ 4*np.abs(predictedAvgVelError)

            #Failsafes cause constraints don't work
            if predictedVelocity < 0.8 or currentVelocity < 0.5:
                tempCostFunction += 25

            if predictedDistFromLeft*predictedDistFromRight > 0:
                tempCostFunction += tempCostFunction*2
                outOfBounds = True

            if (np.abs(predictedDistFromLeft) < 0.75 or np.abs(predictedDistFromRight) < 0.75) and outOfBounds == False:
                tempCostFunction += 100

            currentVelocityX, currentVelocityY = predictedVelocity*np.cos(predictedYaw+beta), predictedVelocity*np.sin(predictedYaw+beta) 
            currentVelocity = predictedVelocity
            currentX, currentY = predictedX, predictedY
            costfn += tempCostFunction*weights[i]

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

        print('cf = ', result.fun)
        print("Number of iterations:", result.nit)
        # print(optimal_values)
        # print(result.message)

        acceleration = optimal_values[0]
        steer = optimal_values[1]
        #print("steering : ",steer)


        self.old_steer = steer
        self.old_acceleration = acceleration      

        return acceleration, steer

    def MainLoop(self):
        
        self.waypoints.IfCrossedWaypoint(self.x,self.y) 
        
        acceleration, steering_angle = self.Calculate()
        self.inputingValue(acceleration,steering_angle)            
                    
        append_values_to_csv([self.x, self.y])

        # if self.v > 1:
        #     self.resetSim(0)

        # if self.show_monitor:
        #     self.update_monitor(acceleration, steering_angle, self.localisation, self.waypoints)

    def LoopSetter(self):
        """
        This function is called when the node is started. It runs the main loop at a fixed rate.
        """
        
        timerPeriod = 1/30
        
        try:
            self.timer = self.create_timer(timerPeriod, self.MainLoop)

        except KeyboardInterrupt:
            print("ROS Interrupt Exception")
            exit(1)

def main(args=None):
    rclpy.init(args=args)
    
    obj = Mpc()

    #obj.show_monitor = False
    
    rclpy.spin(obj)

    obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()