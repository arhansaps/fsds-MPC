#Visualisation
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os.path

from SLAM import Localisation
localisation = Localisation()

x, y = [], []
with open(os.path.join(os.path.dirname(__file__),'conesShifted.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
    plt.scatter(x, y)

cones = []
with open(os.path.join(os.path.dirname(__file__),'conesShifted.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        cones.append([float(row[0]),float(row[1]),row[2],row[3]])
localisation.CreateMap(cones)

for j in range(len(localisation.sectorsBlue)):
    x,y = [], []
    for i in localisation.sectorsBlue[j]:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y)

for j in range(len(localisation.sectorsYellow)):
    x,y = [], []
    for i in localisation.sectorsYellow[j]:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y)

with open(os.path.join(os.path.dirname(__file__), 'wpsNew.csv'), 'r') as file: #waypointposn.csv
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader, start=1):
        x.append(float(row[0]))
        y.append(float(row[1]))
        plt.scatter(float(row[0]), float(row[1]))
        plt.text(float(row[0]), float(row[1]), str(idx), fontsize=9, ha='right')

x, y = [], []
with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
    plt.plot(x, y)

plt.show()

'''
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

csv_file = '/home/gitaash/FM/basicController/src/MPC/temp.csv'
x,y = [],[]
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        #print(row)
        x.append(row[0])
        y.append(row[1])
        if len(x) == 2:
            plt.plot(x,y)
            x,y = [],[]

#x = list(car.X)
#y = list(car.Y)
# Show the graph
#plt.plot(x,y)

plt.show()
'''
