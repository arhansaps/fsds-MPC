import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from scipy.optimize import minimize
import csv


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.get_logger().info('CmdPublisher node started.')
        self.i = -1
        self.max_velocity = 9
        self.time_step = 0.11
        self.previous_error = 0.0
        self.prediction_horizon = 3
        self.smoothing_delta_cost = 0.65
        self.crosstrack_cost = 0.85
        self.heading_cost = 1.0
        self.velocity_cost = 0.5
        self.slip_cost = 0.2
        self.mpc_delta = 0.0
        self.mpc_acceleration=0.0
        self.HE_MEAN=0.2590683478115822
        self.HE_STD =0.7800633223219351
        self.CT_MEAN= 0.3781291695835408
        self.CT_STD =0.3022799890506926
        self.VE_MEAN=  3.0634986204928274
        self.VE_STD= 1.067342332581251
        self.SMT_MEAN=  0.1907798022932672
        self.SMT_STD =0.22289080260575
        self.SLIP_MEAN= 1.3687166249978457
        self.SLIP_STD= 1.0061477979410745
        self.prev_sign_car = -1
        self.prev_sign_pred = -1
        self.L =[
            (4, 0.5),
            (6.0997, 0.46011999999999986),
            (8.958765, 0.6334599999999995),
            (12.904501, 0.5442049999999998),
            (16.50963, 0.9106499999999986),
            (20.142315, 2.0267799999999987),
            (23.03038, 3.0184499999999996),
            (25.728749999999998, 4.120899999999999),
            (27.0522, 3.8519499999999987), 
            (29.24365, 3.0322499999999994),
            (30.2343, 2.159699999999999),
            (31.7488, 0.11077499999999851),
            (32.04075, -1.3419550000000005),
            (31.85275, -4.607830000000001),
            (30.93425, -7.440205000000001),
            (30.10215, -9.54865),
            (28.42945, -11.919970000000001),
            (26.178150000000002, -13.647795),
            (23.266765, -15.095735000000001),
            (20.758225, -16.539845),
            (16.424795, -18.711165),
            (12.720585, -20.563344999999998),
            (9.160635, -22.593),
            (6.340465, -24.4171),
            (3.4438549999999992, -25.2316),
            (2.0123599999999993, -24.977),
            (-0.24815000000000076, -23.59835),
            (-1.1382499999999993, -22.7739),
            (-3.0364499999999985, -20.334455),
            (-3.387450000000001, -16.96707),
            (-3.5478500000000004, -15.764190000000001),
            (-3.2800499999999992, -12.895075),
            (-2.8646499999999993, -10.3),
            (-2.565950000000001, -7.250210000000001),
            (-2.5943500000000004, -4.270505000000001),
            (-1.4700000000000006, -1.7200000000000006),
            (-0.6999999999999993, -0.6400000000000006),
            (0.5199999999999996, -0.26000000000000156),
        ]
        self.subscription = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        yaw = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
        
        velocity = msg.twist.twist.linear
        vel_x = velocity.x
        vel_y = velocity.y


        position = msg.pose.pose.position
        x_pos, y_pos = position.x, position.y
        car_velocity = self.get_velocity(vel_x, vel_y)
        #print(self.i)
        w1 = self.L[self.i%len(self.L)] if self.i!=-1 else self.L[-1]
        w2 = self.L[(self.i+1)%len(self.L)] if self.i!=-1 else self.L[0]
        t = self.check_waypoint_crossed((x_pos, y_pos), w1, w2, self.prev_sign_car)
        if t:
            self.i+=1
            self.prev_sign_car*=-1



        
        U0 = [self.mpc_acceleration, self.mpc_delta] if self.mpc_acceleration!=0 else [3.0, self.mpc_delta]
        toople = (x_pos, y_pos, yaw, car_velocity, self.mpc_delta)
        bounds = [(-1,3), (-0.52,0.52)]
        result = minimize(self.cost_function, U0, args=(toople,), method='SLSQP', bounds=bounds)
        #print("MPC result X: ",result.x)
        self.mpc_acceleration=result.x[0]
        self.mpc_delta=result.x[1]
        pub_msg = AckermannDriveStamped()
        pub_msg.drive.acceleration = self.mpc_acceleration
        pub_msg.drive.steering_angle = self.mpc_delta
        self.publisher_.publish(pub_msg)



    def check_waypoint_crossed(self, cp, w1, w2,sign):
        a = np.array(w1) #PREVIOUS WAYPOINT
        b = np.array(cp) #POSITION
        c = np.array(w2) #CURRENT WAYPOINT
        m = np.arctan((c[1]-a[1])/(c[0]-a[0]))
        m_r=-1/m

        def line(x,y):
            return y-c[1]-m_r*(x-c[0])
        
        answer = line(b[0], b[1])
        ans_sign = 1 if answer>0 else -1
                   
        return True if ans_sign!=sign else False
    def get_velocity(self, vel_x, vel_y):
        r = math.sqrt(vel_x ** 2 + vel_y ** 2)
        return r

    def compute_crosstrack_error(self, cp, wp1, wp2):
        x1, y1 = wp1
        x2, y2 = wp2
        x, y = cp
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        denominator = np.sqrt(A ** 2 + B ** 2) + 1e-6
        return abs(A * x + B * y + C) / denominator



    def compute_heading_error(self, w1, w2):
        dy = w2[1] - w1[1]
        dx = w2[0] - w1[0]
        return (np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi

    def compute_perpendicular_distance(self, xa, ya, w1, w2):
        x1, y1 = w1
        x2, y2 = w2

        # Compute the perpendicular distance to the line
        m = (y2 - y1) / (x2 - x1)
        c = -m * x1 + y1
        d = abs(-m * xa + ya - c) / math.sqrt(m ** 2 + 1)
        return d

    def quaternion_to_euler(self, x, y, z, w):
        # Conversion from quaternion to Euler yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

 

    def get_reference_velocity(self, max_velocity, angle):
        return max_velocity * (1 - abs(angle) / np.pi)

 

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def calc_distance(self, p1, p2):
        return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[0] - p2[0]) ** 2)

    def find_closest_waypoint(self, x, y):
        distances = [(wp[0] - x) ** 2 + (wp[1] - y) ** 2 for wp in self.L]
        sorted_distances = np.argsort(distances)
        first_closest, second_closest = sorted_distances[0], sorted_distances[1]
        print("Closest: ", first_closest)
        print("Second Closest: ", second_closest)
        return max(first_closest, second_closest)

    def cost_function(self, U, t):
        a, delta = U
        x,y,theta,v, current_delta = t
        csv_list = [x,y]
        cost = 0
        weights = np.array([self.crosstrack_cost, self.heading_cost, self.velocity_cost, self.smoothing_delta_cost, self.slip_cost])
        for t in range(self.prediction_horizon):
            wp_idx = self.find_closest_waypoint(x, y)
            wp1 = self.L[wp_idx]
            wp2 = self.L[min(wp_idx + 1, len(self.L) - 1)]
            beta_dyn = math.atan2(math.tan(delta)*0.711, 1.58)
            beta_kin= math.atan2(v*math.sin(delta+theta), v*math.cos(delta+theta))
            cte =abs(self.compute_crosstrack_error([x, y], wp1, wp2))
            he = abs(self.compute_heading_error(wp1, wp2) - theta)
            he = self.normalize_angle(he)
            ve = abs(v - self.get_reference_velocity(self.max_velocity, he))
            smt_e = abs(current_delta-delta)
            ts_e = abs(beta_kin-beta_dyn)

            errors = np.array([abs(cte), abs(he), abs(ve), abs(smt_e), abs(ts_e)])
            cost+=np.dot(weights, errors)
            x += v * math.cos(delta+theta) * self.time_step
            y += v * math.sin(delta+theta) * self.time_step
            delta += (v / 1.58) * math.tan(delta) * self.time_step
            theta += (v / 1.58) * math.tan(delta) * self.time_step
            theta = self.normalize_angle(theta)
            v +=a*self.time_step
     


        return cost
    def normalize_value(self, x,mu, sigma):
        return (x-mu)/sigma if sigma!=0 else (x-mu)



timerPeriod = 1/30


def main(args=None):
    obj = MPCController()
    start = time.time()
    try:
        while True:
            dt = time.time() - start
            if dt >= timerPeriod:
                obj.car_callback()
                start = time.time()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Resetting position...")
        client.reset()
        client.enableApiControl(True)
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