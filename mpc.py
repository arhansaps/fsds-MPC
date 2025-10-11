import rclpy
import math
import numpy as np
import csv
from scipy.optimize import minimize
from rclpy.node import Node
from eufs_msgs.msg import CarState
from ackermann_msgs.msg import AckermannDriveStamped

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.subscriber_ = self.create_subscription(CarState, '/odometry_integration/car_state', self.car_callback, 10)
        
        self.N = 5 #horizon, was 12
        self.dt = 0.1 #timestep
        self.Lf = 1.58/2  # distance from center of mass to front axle
        
        # Reference values
        self.ref_v = 6.0
        
        self.w_cte = 30 #was 35   #cte error ka weight
        self.w_epsi = 30   #was 35.5  #heading error ka weight
        self.w_v = 10  #was 18   #velocity weight
        self.w_delta = 23  #steering angle weight
        self.w_a = 5        #acc weight
        
        self.path = [  
            (4, 0.5), (6.1, 0.46), (8.96, 0.63), (12.9, 0.54), (16.5, 0.91), (20.14, 2.02),
            (23.03, 3.01), (25.72, 4.12), (27.05, 3.85), (29.24, 3.03), (30.23, 2.15),
            (31.74, 0.11), (32.04, -1.34), (31.85, -4.60), (30.93, -7.44), (30.10, -9.54),
            (28.42, -11.91), (26.17, -13.64), (23.26, -15.09), (20.75, -16.53),
            (16.42, -18.71), (12.72, -20.56), (9.16, -22.59), (6.34, -24.41),
            (3.44, -25.23), (2.01, -24.97), (-0.24, -23.59), (-1.13, -22.77),
            (-3.03, -20.33), (-3.38, -16.96), (-3.54, -15.76), (-3.28, -12.89),
            (-2.86, -10.3), (-2.56, -7.25), (-2.59, -4.27), (-1.47, -1.72),
            (-0.7, -0.64)
        ]

        self.current_waypoint_idx = 0
        self.ref_path_x = np.array([point[0] for point in self.path])
        self.ref_path_y = np.array([point[1] for point in self.path])
        
        # Initialize CSV file
        self.csv_file = open('mpc_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['car_x', 'car_y', 'next_point_x', 'next_point_y'])
        
        self.get_logger().info('MPC Controller initialized')

    def car_callback(self, msg):
        self.msg = msg
        self.run_mpc()

    def advance_waypoint(self, x, y, waypoint_idx):
        path_length = len(self.path)
        while True:
            current_idx = waypoint_idx % path_length #current waypt
            next_idx = (waypoint_idx + 1) % path_length #for wrap around path loop
            
            A = self.path[(current_idx - 1) % path_length] #previous point
            B = (x, y) #current pos
            C = self.path[current_idx] #target waypt
            
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
        return self.path[self.current_waypoint_idx]

    def run_mpc(self):
        px = self.msg.pose.pose.position.x
        py = self.msg.pose.pose.position.y
        q = self.msg.pose.pose.orientation
        psi = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v = math.sqrt((self.msg.twist.twist.linear.x ** 2) + (self.msg.twist.twist.linear.y ** 2))

        ref_x, ref_y = self.next_points(px, py, psi) #target waypt

        # Write to CSV file
        self.csv_writer.writerow([px, py, ref_x, ref_y])
        self.csv_file.flush()

        target_x = ref_x
        target_y = ref_y

        # Calculate theta (angle between car heading and path direction) #to calculate heading error
        path_angle = math.atan2(target_y - py, target_x - px) 
        theta = abs(psi - path_angle)
        
        # normalize angle
        while theta > math.pi:
            theta -= 2 * math.pi
        theta = abs(theta)
        if theta > math.pi/2:
            theta = math.pi - theta
            
        dynamic_velocity=self.ref_v*((math.cos(theta)**(4/1)))  #use costheta for descreasing speed when theta increases

        cte = math.sqrt((px - target_x)**2 + (py - target_y)**2)
        heading_error = math.atan2(target_y - py, target_x - px) - psi

        #changes made here
        while heading_error > math.pi:
            heading_error -= 2*math.pi
        while heading_error < math.pi:
            heading_error += 2 * math.pi

        state = [px, py, psi, v, cte, heading_error, dynamic_velocity]

        solution = self.solve_mpc(state)

        steer_value = solution[0]
        throttle_value = solution[1]

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = float(steer_value)
        drive_msg.drive.acceleration = float(throttle_value)
        self.publisher_.publish(drive_msg)
        self.get_logger().info(f"MPC: steering={steer_value:.3f}, throttle={throttle_value:.3f}, cte={cte:.3f}, v={v:.3f}, dynamic_v={dynamic_velocity:.3f}, angle={theta:.3f}")

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
                ref_x, ref_y = self.path[local_waypoint_idx % len(self.path)]

                cte = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                epsi = math.atan2(ref_y - y, ref_x - x) - psi

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
            self.get_logger().warn(f"MPC optimization failed: {result.message}")
            return [0.0, 0.0]

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    
    mpc_controller = MPCController()
    
    try:
        rclpy.spin(mpc_controller)
    except KeyboardInterrupt:
        pass
    finally:
        mpc_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()