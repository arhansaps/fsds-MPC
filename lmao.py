import sys
import os
import time
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("127.0.0.1")

client.confirmConnection()

client.enableApiControl(True)
def acceleration_to_throttle_brake(acc, a_max=1.0, b_max=1.0):
        acc = np.clip(acc, -b_max, a_max)

        if acc >= 0:
            throttle = acc / a_max
            brake = 0.0
            throttle = np.clip(throttle,0.0,0.2)
        else:
            throttle = 0.0
            brake = -acc / b_max
            #brake = 1
            brake = np.clip(brake,0.0,0.5)
        return throttle, brake 

max_throttle = 0.2 # m/s^2
target_speed = 4 # m/s
max_steering = 0.3
cones_range_cutoff = 7 # meters
df = pd.read_csv("//home//hackysapy//Desktop//FMDV//FSDS_fm//src//Control//MPC//monitor_log.csv")

min_accel = df["Acceleration (m/s^2)"].min()
max_accel = df["Acceleration (m/s^2)"].max()

def main():
    with open("//home//hackysapy//Desktop//FMDV//FSDS_fm//src//Control//MPC//monitor_log.csv") as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            
            accel = float(row["Acceleration (m/s^2)"])
            steering = float(row["Steering Angle (rad)"])
            #velocity = float(row['Velocity (m/s)'])
            gps = client.getGpsData()
            velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))
            #acceleration = (accel - min_accel) / (max_accel - min_accel)
            throttle = max_throttle * max(1 - velocity / target_speed, 0)
            car_controls = fsds.CarControls()
            acceleration, brake = acceleration_to_throttle_brake(accel)
            print("Accel:",acceleration)
            print("Throttle:",throttle)
            print("Steering",steering)
            #throttle, brake = acceleration_to_throttle_brake(accel)
            if(i<1):
                car_controls.throttle = 0.2
            else:
                car_controls.throttle = car_controls.throttle - 0.02
                if(car_controls.throttle < 0):
                    car_controls.throttle = car_controls.throttle * -1
            
            i+=1
            print(i)
            print(velocity)
            car_controls.brake = 0
            car_controls.steering = -1*steering 
            client.setCarControls(car_controls)

            time.sleep(0.2)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()