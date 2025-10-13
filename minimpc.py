import rclpy
from scipy.optimize import minimize
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped as control
import numpy as np
import csv
from nav_msgs.msg import Odometry as pos
import types
from casadi import *
import casadi as ca
from pointmodel import pointmodel
class mpc(Node):
    def __init__(self):
        super().__init__("minimpc")
        self.waypoints = []
        with open("waypoints","r") as file:
            read = csv.reader(file)
            for i in read:
                self.waypoints.append([float(i[0]),float(i[1])])
        self.model = pointmodel()
        #path data
        self.m         =  self.m_perp      = self.theta        =  self.prev_op_sign        =  self.wp_index  =  0
        self.theta_MPC =  self.m_perp_MPC  =  self.prev_op_MPC = self.yaw_MPC              =   0  #MPC prediction data
        
        #lap timer data
        self.lap_time_start =  None  
        self.total_laps     =  0 

        #car data 
        self.ControlVector      = [1,0] 
        self.x          = self.y  =  self.yaw = 0
        self.vel        = [0,0]
        self.WheelBase  = 1.580 #m

        #ROS2 objects
        self.publisher = self.create_publisher(control, '/cmd', 10)
        self.car       = control()
        self.pos       = self.create_subscription(pos,'ground_truth/odom',self.ControlVector_update,10)
        self.publisher.publish(self.car)
    def Errors(self):
        """
        Makes casadii error terms
        returns: cross track error,heading error,velocity error
        """
        errorTerms = types.SimpleNamespace()

        errorTerms.cross_track_error = self.cross_track_error(self.x, self.y, self.wp_index, self.waypoints)
        errorTerms.heading_error = self.heading_error_ca(self.x, self.y, self.theta)
        errorTerms.velocity_error = self.vel_error(self.theta, self.vel, self.wp_index)
        return errorTerms
    
    def ControlVector_update(self,msg): #callback function for updating the ControlVector of the car
       
        q          =   msg.pose.pose.orientation #quarternion data
        t3         =   +2.0 * (q.w * q.z + q.x * q.y)
        t4         =   +1.0 - 2.0 * (q.y*q.y + q.z * q.z)
        self.yaw   =  np.arctan2(t3, t4)
        self.vel   =   msg.twist.twist.linear
        self.theta =   self.straight_line(self.wp_index)
        self.x     =  msg.pose.pose.position.x
        self.y     =  msg.pose.pose.position.y
        self.update_waypoint(self.x,self.y)
        self.model.state = [self.x,self.y,self.theta,self.vel,self.wp_index]

        
        self.optimize_control()
        return None
    
    def global_to_car_frame(self,carpos:list,point:list,yaw:int)->list:
        '''
        args: 
            carpos ->position of the car in cartisian
            point ->positio of the point in cartisian
            yaw -> yaw of the car

        returns:
            transformed point in car frame  
        '''
        """
        transform = [point - car]R(theta) first subtract the vector then rotate
        """
        rotation_matrix = np.array([
            [np.cos(-yaw),-np.sin(-yaw)]
            [np.sin(-yaw),np.cos(-yaw)]
        ])
        linearTransform = vertcat(point[0]-carpos[0],point[1]-carpos[1])
        transform = rotation_matrix @ linearTransform 
        return transform[0],transform[1]
    


    def update_waypoint_predict_casadi(self,waypoints, wp_index, prev_op, x, y):
        """
        CasADi function to update the waypoint index and operation sign.
        """
        # Define CasADi variables
        wp_index = ca.MX(wp_index)
        prev_op = ca.MX(prev_op)
        x = ca.MX(x)
        y = ca.MX(y)
        waypoints = ca.MX(waypoints)

        # Get the current waypoint coordinates
        x1 = waypoints[wp_index % waypoints.size1(), 0]
        y1 = waypoints[wp_index % waypoints.size1(), 1]

        # Get the next waypoint coordinates
        x2 = waypoints[(wp_index + 1) % waypoints.size1(), 0]
        y2 = waypoints[(wp_index + 1) % waypoints.size1(), 1]

        # Calculate the slope and perpendicular slope
        m = (y2 - y1) / (x2 - x1)
        m_perp = -1 / m

        # Calculate the perpendicular distance from the waypoint line
        c = y1 - (m_perp * x1)
        op = m_perp * x + c - y

        # Check if the sign of the perpendicular distance has changed
        wp_index_updated = ca.if_else((prev_op > 0 and op < 0) or (prev_op < 0 and op > 0) or op == 0, wp_index + 1, wp_index)

        # Update the previous perpendicular distance sign
        prev_op_updated = op

        # Create CasADi function
        f = ca.Function('update_waypoint_predict', [waypoints, wp_index, prev_op, x, y], [wp_index_updated, prev_op_updated])

        return f

    def heading_error_ca(self):
        """
        Create a CasADi function to calculate the heading error by transforming the coordinates to the car frame.
        """
        # Define CasADi symbols
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        wp_index = ca.SX.sym('wp_index', 1)
        waypoints = ca.SX.sym('waypoints', self.waypoints.shape[0], self.waypoints.shape[1])

        # Get target waypoint
        target_x = waypoints[wp_index, 0]
        target_y = waypoints[wp_index, 1]

        # Define car position and target point
        carpos_sym = ca.vertcat(x, y)
        point_sym = ca.vertcat(target_x, target_y)
        yaw_sym = theta

        # Rotation matrix
        rotation_matrix = ca.vertcat(
            ca.horzcat(ca.cos(-yaw_sym), -ca.sin(-yaw_sym)),
            ca.horzcat(ca.sin(-yaw_sym), ca.cos(-yaw_sym))
        )

        # Linear transformation
        linear_transform = ca.vertcat(point_sym[0] - carpos_sym[0], point_sym[1] - carpos_sym[1])

        # Transform
        transform = ca.mtimes(rotation_matrix, linear_transform)

        # Calculate the heading error
        heading_error = ca.atan2(transform[1], transform[0])

        # Create and return CasADi function
        return ca.Function('heading_error', [x, y, theta, wp_index, waypoints], [heading_error])
        

    def vel_error(theta, v, wp_index):
        # Define CasADi symbols
        theta_sym = ca.SX.sym('theta')
        v_sym = ca.SX.sym('v')
        wp_index_sym = ca.SX.sym('wp_index', 1)

        # Define the reference velocity
        ref_x_dot = ca.if_else(
            ca.logic_or(wp_index_sym == 0, v_sym < 0),
            2.5,  # arbitrary value
            8 * ca.cos(theta_sym)
        )

        # Calculate the velocity error
        vel_error = ca.fabs(ref_x_dot - v_sym)

        # Create CasADi function
        vel_error_func = ca.Function('vel_error', [theta_sym, v_sym, wp_index_sym], [vel_error])

        return vel_error_func(theta, v, wp_index)
    def cross_track_error(x, y, wp_index, waypoints, k=1):
        # Define CasADi symbols
        x_sym = ca.SX.sym('x')
        y_sym = ca.SX.sym('y')
        wp_index_sym = ca.SX.sym('wp_index', 1)
        waypoints_sym = ca.SX.sym('waypoints', waypoints.shape[0], waypoints.shape[1])
        k_sym = ca.SX.sym('k', 1)

        # Extract waypoints
        x0 = waypoints_sym[(wp_index_sym - k_sym) % waypoints.shape[0], 0]
        y0 = waypoints_sym[(wp_index_sym - k_sym) % waypoints.shape[0], 1]
        x1 = waypoints_sym[(wp_index_sym - 0) % waypoints.shape[0], 0]
        y1 = waypoints_sym[(wp_index_sym - 0) % waypoints.shape[0], 1]
        x2 = waypoints_sym[(wp_index_sym + k_sym) % waypoints.shape[0], 0]
        y2 = waypoints_sym[(wp_index_sym + k_sym) % waypoints.shape[0], 1]

        # Calculate line equation coefficients
        A = -(y2 - y1)  # Ax + By = C (straight line std form)
        B = (x2 - x1)
        C = -(y1 - y0)
        D = (x1 - x0)

        # Calculate cross track errors
        c0 = ca.fabs((D * (y0 - y_sym) - C * (x0 - x_sym)) / ca.sqrt(C**2 + D**2))
        c1 = ca.fabs((B * (y1 - y_sym) - A * (x1 - x_sym)) / ca.sqrt(A**2 + B**2))
        c = 0.3 * c0 + 0.7 * c1

        # Create CasADi function
        cross_track_error_func = ca.Function('cross_track_error', [x_sym, y_sym, wp_index_sym, waypoints_sym, k_sym], [c])

        return cross_track_error_func(x, y, wp_index, waypoints, k)
    def cost_function(self, model):
        # Convert waypoints to CasADi array
        waypoints_ca = ca.DM(self.waypoints)
        # Define CasADi symbols for state and control
        x = model.state[0]
        y = model.state[1]
        theta = model.state[2]
        v = model.state[3]
        a = model.control[0]
        delta = model.control[1]
        
        # Define waypoint index and other necessary symbols
        wp_index = ca.SX.sym('wp_index', 1)
        m_perp_MPC = ca.SX.sym('m_perp_MPC')
        prev_op_MPC = ca.SX.sym('prev_op_MPC')

        # Calculate cost terms
        cross_track_error = self.cross_track_error(x, y, wp_index, waypoints_ca)
        heading_error = self.heading_error_ca(x, y, theta)
        velocity_error = self.vel_error(theta, v, wp_index)

        # Define weights as CasADi symbols
        w_cte = ca.SX.sym('w_cte')
        w_he = ca.SX.sym('w_he')
        w_ve = ca.SX.sym('w_ve')

        # Total cost
        cost = w_cte * cross_track_error + w_he * heading_error + w_ve * velocity_error 

        # Update waypoint index
        wp_index_next, prev_op_MPC_new = self.update_waypoint_predict(
            x, y, wp_index, waypoints_ca, m_perp_MPC, prev_op_MPC
        )

        # Create and return CasADi function
        return ca.Function('cost_function', 
                        [model.state, model.control, wp_index, m_perp_MPC, prev_op_MPC, w_cte, w_he, w_ve, w_ce], 
                        [cost, wp_index_next, prev_op_MPC_new])
