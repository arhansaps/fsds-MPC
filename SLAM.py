import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os.path

import csv

class Localisation():
    def __init__(self):
        self.sectorRangeThreshold = 5
        self.sectorsBlue, self.sectorsYellow = [], []
        self.prevArrCones = []
        self.numOfSectors = 0

        #Cache of sorts
        #self.prevNearestBlueCones, self.prevNearestYellowCones = [], []

        #Saves curvature foer every new set of 4+ cones
        self.allPrevSplinesBlue, self.allPrevSplinesYellow = {}, {}
        self.prevCurvatureBlue, self.prevCurvatureYellow = 0,0
        self.prevXSC, self.prevYSC = 0,0

        #Abi autsim
        # self.xOldYellowCone = []
        # self.yOldYellowCone = []

        # #Cache for get_MengerCurvature
        # self.allPrevCurvaturesBlue = {}
        # self.allPrevCurvaturesYellow = {}
        # self.prevX, self.prevY = 0,0
        # self.prevCurvatureBlue, self.prevCurvatureYellow = 0,0

        # #Only for the monitor, its a list of x,y of each waypoint. Can be done here also
        # Waypoints = pd.read_csv(os.path.join(os.path.dirname(__file__),"waypointposn.csv"))
        # self.waypoints = []
        # for i in range(1, len(Waypoints)):
        #     self.waypoints.append(list(Waypoints.iloc[i,:]))

    def ShortestDistance(self, x2, y2, x1, y1, x, y):
        delx = x2 - x1
        dely = y2 - y1
        A = dely/delx
        B = -1
        C = -1*(dely/delx)*x1 + y1
        d = (A*x + B*y + C)/np.sqrt(A**2 + B**2)
        return d    
    
    def Distance(self, x2, y2, x1, y1):
        d = np.sqrt((x2 - x1)**2 + (y2-y1)**2)
        return d  
    
    def NearestCones(self, x, y): #Might need to consider orange cones - not needed as of now
        sectorIndex = int(np.sqrt(x**2 + y**2)/self.sectorRangeThreshold)

        if sectorIndex > self.numOfSectors-1:
            sectorIndex = self.numOfSectors-1
        if sectorIndex == self.numOfSectors-1:  #As there are only n+1 sectors, starting from 0,1,2,3,4,5,6..n
            sectorIndex -= 1
        PreviousSectorIndex = sectorIndex - 1

        numBlueSec = min(3, len(self.sectorsBlue))

        Temp_Dict = {}
        for i in range(numBlueSec):     #Taking prev, current and the next sector

            for cone in self.sectorsBlue[PreviousSectorIndex+i]:
                dist = np.sqrt((x-cone[0])**2 + (y-cone[1])**2)
                Temp_Dict[dist]=cone
        SortedDistance = sorted(Temp_Dict.keys())
        nearestBlueCones = []
        for i in SortedDistance:
            nearestBlueCones.append(tuple(Temp_Dict[i]))

        numYellowSec = min(3,len(self.sectorsYellow))
        Temp_Dict = {}
        for i in range(numYellowSec):
            for cone in self.sectorsYellow[PreviousSectorIndex+i]:
                dist = np.sqrt((x-cone[0])**2 + (y-cone[1])**2)
                Temp_Dict[dist]=cone
        SortedDistance = sorted(Temp_Dict.keys())
        NearestYellowCones = []
        for i in SortedDistance:
            NearestYellowCones.append(tuple(Temp_Dict[i]))
        return nearestBlueCones,NearestYellowCones


    def DistanceFromSides(self, x, y):
        nearestBlueCones, NearestYellowCones = self.NearestCones(x, y)      
        DistanceFromLeft = self.ShortestDistance(nearestBlueCones[1][0],nearestBlueCones[1][1],nearestBlueCones[0][0],nearestBlueCones[0][1],x,y)
        DistanceFromRight = self.ShortestDistance(NearestYellowCones[1][0],NearestYellowCones[1][1],NearestYellowCones[0][0],NearestYellowCones[0][1],x,y)
        (#x1 = [nearestBlueCones[1][0],nearestBlueCones[0][0]]
        #y1 = [nearestBlueCones[1][1],nearestBlueCones[0][1]]
        #x2 = [NearestYellowCones[1][0], NearestYellowCones[0][0]]
        #y2 = [NearestYellowCones[1][1],NearestYellowCones[0][1]]
        #Ignore, For visualisation
        #if self.Current_sector == 1000:
        #    for j in range(7):
        #        x,y = [], []
        #        for i in self.sectorsBlue[j]:
        #            x.append(i[0])
        #            y.append(i[1])
        #        for i in self.sectorsYellow[j]:
        #            x.append(i[0])
        #            y.append(i[1])
        #        plt.scatter(x,y)
        #    plt.plot(x1,y1)
        #    plt.plot(x2,y2)
        #    plt.show()
        )
        return DistanceFromLeft, DistanceFromRight

    def IsSame(self, cone1, cone2):
        dist = np.sqrt((cone1[1]- cone2[1])**2 + (cone1[2] - cone2[2])**2)
        if dist < 1.5:
            return True
        else:
            return False
        
    def GetWeightedAvg(self, cone1, cone2):
        w1, w2 = 0.7, 0.3
        w_avg = []
        w_avg.append(cone2[0]*w1 + cone1[0]*w2)
        w_avg.append(cone2[1]*w1 + cone1[1]*w2)
        return w_avg

    def CreateMap(self,cones):

        if cones == self.prevArrCones:
            print("No new cones found")
            return
        
        self.prevArrCones = cones

        sectorsBlue = self.sectorsBlue
        sectorsYellow = self.sectorsYellow

        for cone in cones:
            color = cone[0]
            x = float(cone[1])
            y = float(cone[2])

            coneRange = np.sqrt((x**2) + (y**2))
            sectorIndex = int(coneRange // self.sectorRangeThreshold)

            if (color == "blue"):
                sectors = sectorsBlue
            elif (color=='yellow'):
                sectors = sectorsYellow
            else:
                continue

            while (sectorIndex > len(sectors)-1):
                sectors.append([])
            
            sect = sectors[sectorIndex]
            sameFound = 0
            for j in range(len(sect)):
                if self.IsSame(cone,sect[j]):
                    sameFound = 1
                    sect[j] = self.GetWeightedAvg(cone,sect[j])
                    break

            if sameFound == 0:
                sect.append(cone)

        self.sectorsBlue = sectorsBlue
        self.sectorsYellow = sectorsYellow

        if self.sectorsYellow > self.sectorsBlue:
            self.numOfSectors = len(self.sectorsYellow)
        else:
            self.numOfSectors = len(self.sectorsBlue)

    def ConvertToCarFrame(self,car, x, y):
        relative_posn= -np.array([car.x,car.y])+np.array([x,y])
        rotation_matrix=np.array([
            [np.cos(-car.yaw), -np.sin(-car.yaw)],
            [np.sin(-car.yaw), np.cos(-car.yaw)]]
            )
        posnInCarFrame =rotation_matrix @ relative_posn
        return posnInCarFrame

    ('''def get_MengerCurvature(self, x, y):
        if x == self.prevX and y == self.prevY:
            return self.prevCurvatureBlue, self.prevCurvatureYellow
        curvatureBlue, curvatureYellow = None, None

        nearestBlueCones, NearestYellowCones = self.NearestCones(x, y)
        nearestBlueCones, NearestYellowCones = tuple(nearestBlueCones[0:3]), tuple(NearestYellowCones[0:3])

        if nearestBlueCones in list(self.allPrevCurvaturesBlue.keys()):
            curvatureBlue = self.allPrevCurvaturesBlue[nearestBlueCones]

        if NearestYellowCones in list(self.allPrevCurvaturesYellow.keys()):
            curvatureYellow = self.allPrevCurvaturesYellow[NearestYellowCones]

        def get_curvature(cones):
            AB = self.Distance(cones[1][0], cones[1][1], cones[0][0], cones[0][1])
            BC = self.Distance(cones[2][0], cones[2][1], cones[1][0], cones[1][1])
            CA = self.Distance(cones[0][0], cones[0][1], cones[2][0], cones[2][1])
            A = np.abs(0.5*((cones[1][0]-cones[0][0])*(cones[2][1]-cones[0][1])-(cones[2][0]-cones[0][0])*(cones[1][1]-cones[0][1])))
            return np.abs((4*A)/(AB*BC*CA))
        
        if curvatureBlue is None:
            curvatureBlue = get_curvature(nearestBlueCones)
            self.allPrevCurvaturesBlue[nearestBlueCones] = curvatureBlue
        if curvatureYellow is None:
            curvatureYellow = get_curvature(NearestYellowCones)
            self.allPrevCurvaturesYellow[NearestYellowCones] = curvatureYellow
        
        WcurvatureBlue, WcurvatureYellow = np.abs(curvatureBlue), np.abs(curvatureYellow)   #W -> weighted
        #if WcurvatureBlue > WcurvatureYellow:     #Increasing weight on one side
        #    WcurvatureBlue *= 2
        #else:
        #    WcurvatureYellow *= 2
        
        self.prevX, self.prevY = x,y
        self.prevCurvatureBlue, self.prevCurvatureYellow = WcurvatureBlue, WcurvatureYellow

        return WcurvatureBlue, WcurvatureYellow''')

    def GetSplineAndCurvature(self,  x, y, car=None):
        curvatureBlue, curvatureYellow = None, None
        splineBlue, splineYellow = None, None

        nearestBlueCones, nearestYellowCones = self.NearestCones(x, y)
        nearestBlueCones, NearestYellowCones = tuple(nearestBlueCones[0:4]), tuple(NearestYellowCones[0:4])
        
        def GetSpline(cones_x,cones_y):
            spline=CubicSpline(cones_x,cones_y)
            return spline
        
        def GetCurvature(spline, NearestCones):
            first_derivative = spline.derivative(1)(NearestCones[0][0])
            second_derivative = spline.derivative(2)(NearestCones[0][0])#cones_x[0])
            curvature = second_derivative / (1 + first_derivative**2)**1.5
            return np.abs(curvature)

        if nearestBlueCones in list(self.allPrevSplinesBlue.keys()):
            if x == self.prevXSC and y == self.prevYSC:
                curvatureBlue = self.prevCurvatureBlue
            else:
                curvatureBlue = GetCurvature(self.allPrevSplinesBlue[nearestBlueCones], nearestBlueCones) #Saving all splines in the dict

        if NearestYellowCones in list(self.allPrevSplinesYellow.keys()):
            if x == self.prevXSC and y == self.prevYSC:
                curvatureYellow = self.prevCurvatureYellow
            else:
                curvatureYellow = GetCurvature(self.allPrevSplinesYellow[NearestYellowCones], NearestYellowCones)
            #curvatureYellow = self.allPrevCurvaturesYellow[NearestYellowCones]        
        
        if curvatureBlue is not None and curvatureYellow is not None:
            return curvatureBlue, curvatureYellow

        if len(nearestBlueCones) < 4 or len(NearestYellowCones) < 4:
            print("Not enough cones")
        blue_x,blue_y,yellow_x,yellow_y = [],[],[],[]            

        if curvatureBlue is None:
            nearestBlueConesSorted = sorted(nearestBlueCones)
            for i in range(4):
                blue_x.append(nearestBlueConesSorted[i][0])
                blue_y.append(nearestBlueConesSorted[i][1])
            splineBlue = GetSpline(blue_x, blue_y)
            curvatureBlue = GetCurvature(splineBlue, nearestBlueCones)

        if curvatureYellow is None:
            NearestYellowConesSorted = sorted(NearestYellowCones)
            for i in range(4):
                yellow_x.append(NearestYellowConesSorted[i][0])
                yellow_y.append(NearestYellowConesSorted[i][1])
            splineYellow = GetSpline(yellow_x, yellow_y)
            curvatureYellow = GetCurvature(splineYellow, NearestYellowCones)

            (# self.xYellowCone, self.yYellowCone = yellow_x, yellow_y
            # #Abi Autism
            # if self.xOldYellowCone != self.xYellowCone and self.yOldYellowCone != self.yYellowCone: 
            #     with open('temp.csv', 'a', newline='') as csvfile:
            #         spamwriter = csv.writer(csvfile)
            #         for x in range(0,4):
            #             spamwriter.writerow([self.xYellowCone[x],self.yYellowCone[x]])
            #     self.xOldYellowCone = self.xYellowCone
            #     self.yOldYellowCone = self.yYellowCone
            )


        #curvatureBlue > curvatureYellow     #=> Blue is on the inside of the turn
        # if curvatureBlue > curvatureYellow:
        #     curvatureBlue *= 1.5
        # else:
        #     curvatureYellow *= 1.5
        
        if splineBlue is None:
            splineBlue = self.allPrevSplinesBlue[nearestBlueCones]
        if splineYellow is None:
            splineYellow = self.allPrevSplinesYellow[NearestYellowCones]
        self.allPrevSplinesBlue[nearestBlueCones] = splineBlue
        self.allPrevSplinesYellow[NearestYellowCones] = splineYellow

        self.prevCurvatureBlue, self.prevCurvatureBlue = curvatureBlue, curvatureYellow

        self.prevXSC, self.prevYSC = x, y
        return curvatureBlue, curvatureYellow

    def getNeighbouringCones(self, xCheck):                     #Official code - only for the monitor
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
                            if self.IsSame(sect[j], arrCones[k]):
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
    
'''
import functools
class Localisation:
    # ... other methods ...

    @lru_cache(maxsize=128)  # Adjust cache size as needed
    def get_spline_and_curvature_cached(self, x, y):
        return self.get_spline_and_curvature(x, y)

Wherever you were using localisation.get_spline_and_curvature, replace it with localisation.get_spline_and_curvature_cached
'''