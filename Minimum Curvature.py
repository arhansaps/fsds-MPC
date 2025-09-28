import numpy as np
import csv
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class PathOptimizer:
    def __init__(self):
        self.smoothing_factor = 0.7
        self.min_distance = 0.15
        self.car_width = 1.58  
        self.safety_margin = 0.2  
        
    def load_track_data(self, data_string):
        """Parse the CSV data into waypoints and cones"""
        lines = data_string.strip().split('\n')
        waypoints = []
        yellow_cones = []
        blue_cones = []
        orange_cones = []
        
        for line in lines:
            try:
                x, y, type_label = line.split(',')
                x, y = float(x), float(y)
            except ValueError:
                print(f"Error: Unable to parse line: {line}")
                continue
            
            if 'Waypoint' in type_label:
                waypoints.append([x, y])
            elif 'Yellow Cone' in type_label:
                yellow_cones.append([x, y])
            elif 'Blue Cone' in type_label:
                blue_cones.append([x, y])
            elif 'Orange Cone' in type_label:
                orange_cones.append([x, y])
                
        if not waypoints:
            print("Error: No waypoints found in the data.")
            return None, None, None, None
        return np.array(waypoints), np.array(yellow_cones), np.array(blue_cones), np.array(orange_cones)
    
    def calculate_track_width(self, yellow_cones, blue_cones):
        """Calculate average track width using nearest cone pairs"""
        if len(yellow_cones) == 0 or len(blue_cones) == 0:
            print("Error: Not enough cone data to calculate track width.")
            return None
        
        track_widths = []
        for yellow in yellow_cones:
            distances = np.linalg.norm(blue_cones - yellow, axis=1)
            nearest_blue = np.min(distances)
            track_widths.append(nearest_blue)
        
        if len(track_widths) == 0:
            print("Error: Unable to calculate track width.")
            return None
        
        return np.mean(track_widths)

    def get_nearest_cones(self, point, yellow_cones, blue_cones):
        """Find nearest yellow and blue cones to a point"""
        if len(yellow_cones) == 0 or len(blue_cones) == 0:
            return None, None
            
        yellow_distances = np.linalg.norm(yellow_cones - point, axis=1)
        blue_distances = np.linalg.norm(blue_cones - point, axis=1)
        
        nearest_yellow = yellow_cones[np.argmin(yellow_distances)]
        nearest_blue = blue_cones[np.argmin(blue_distances)]
        
        return nearest_yellow, nearest_blue

    def check_clearance(self, point, yellow_cones, blue_cones):
        """Check if a point maintains sufficient clearance from cones"""
        required_clearance = (self.car_width / 2) + self.safety_margin
        
        nearest_yellow, nearest_blue = self.get_nearest_cones(point, yellow_cones, blue_cones)
        if nearest_yellow is None or nearest_blue is None:
            return True  # If no cones found, assume clearance is OK
            
        yellow_distance = np.linalg.norm(point - nearest_yellow)
        blue_distance = np.linalg.norm(point - nearest_blue)
        
        return yellow_distance >= required_clearance and blue_distance >= required_clearance

    def optimize_path(self, waypoints, track_width, yellow_cones, blue_cones):
        """Optimize racing line for minimum curvature with clearance constraints"""
        if waypoints is None or len(waypoints) < 2:
            print("Error: Not enough waypoints to optimize path.")
            return None

        points = np.array(waypoints)

        # Fit initial spline
        tck, u = splprep([points[:,0], points[:,1]], s=self.smoothing_factor, per=True)

        # Generate more points for optimization
        u_new = np.linspace(0, 1, len(waypoints) * 2)
        x_spline, y_spline = splev(u_new, tck)

        # Calculate curvature at each point
        dx = np.gradient(x_spline)
        dy = np.gradient(y_spline)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5

        # Weight points based on curvature
        weights = 1 / (1 + curvature)
        
        adjusted_points = []
        max_iterations = 150  # Prevent infinite loops
        step_size = 0.2  
        
        for i in range(len(x_spline)):
            point = np.array([x_spline[i], y_spline[i]])
            iteration = 0
            
            while not self.check_clearance(point, yellow_cones, blue_cones) and iteration < max_iterations:
                nearest_yellow, nearest_blue = self.get_nearest_cones(point, yellow_cones, blue_cones)
                
                if nearest_yellow is None or nearest_blue is None:
                    break
                    
                # Calculate middle point between nearest cones
                middle = (nearest_yellow + nearest_blue) / 2
                
                # Calculate vector from point to middle
                direction = middle - point
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    # Normalize and scale the movement
                    normalized_direction = direction / direction_norm
                    # Move point towards middle by step_size
                    point = point + normalized_direction * step_size
                    
                iteration += 1
            
            adjusted_points.append(point)

        adjusted_points = np.array(adjusted_points)

        # Fit final spline with adjusted points
        tck, u = splprep([adjusted_points[:,0], adjusted_points[:,1]], w=weights, s=self.smoothing_factor*len(weights), per=True)

        # Generate final smooth path
        u_final = np.linspace(0, 1, len(waypoints))
        x_final, y_final = splev(u_final, tck)

        return np.column_stack((x_final, y_final))


    def visualize_track(self, waypoints, yellow_cones, blue_cones, orange_cones, optimized_path):
        """Visualize the track with car width boundaries"""
        plt.figure(figsize=(12, 12))
        
        # Plot cones
        if len(yellow_cones) > 0:
            plt.scatter(yellow_cones[:,0], yellow_cones[:,1], c='yellow', marker='^', label='Yellow Cones')
        if len(blue_cones) > 0:
            plt.scatter(blue_cones[:,0], blue_cones[:,1], c='blue', marker='^', label='Blue Cones')
        if len(orange_cones) > 0:
            plt.scatter(orange_cones[:,0], orange_cones[:,1], c='orange', marker='^', label='Orange Cones')

        # Plot original waypoints
        waypoints = np.array(waypoints)
        plt.plot(waypoints[:,0], waypoints[:,1], 'r--', label='Original Path')

        # Plot optimized path
        optimized_path = np.array(optimized_path)
        plt.plot(optimized_path[:,0], optimized_path[:,1], 'g-', linewidth=2, label='Optimized Path')
        
        # Plot car width boundaries around optimized path
        car_half_width = self.car_width / 2
        for i in range(len(optimized_path) - 1):
            p1 = optimized_path[i]
            p2 = optimized_path[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:
                nx = -dy/norm * car_half_width
                ny = dx/norm * car_half_width
                plt.plot([p1[0] + nx, p2[0] + nx], [p1[1] + ny, p2[1] + ny], 'k:', alpha=0.3)
                plt.plot([p1[0] - nx, p2[0] - nx], [p1[1] - ny, p2[1] - ny], 'k:', alpha=0.3)
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Track Layout with Car Width Boundaries')
        plt.show()

def main(csv_file_path):
    data_string = ''
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                data_string += ','.join(row) + '\n'
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return None

    optimizer = PathOptimizer()
    
    waypoints, yellow_cones, blue_cones, orange_cones = optimizer.load_track_data(data_string)
    if waypoints is None:
        print("Terminating due to missing or malformed data.")
        return None
    
    track_width = optimizer.calculate_track_width(yellow_cones, blue_cones)
    if track_width is None:
        print("Terminating due to calculation errors.")
        return None

    optimized_path = optimizer.optimize_path(waypoints, track_width, yellow_cones, blue_cones)
    if optimized_path is None:
        print("Terminating due to optimization failure.")
        return None
    
    optimizer.visualize_track(waypoints, yellow_cones, blue_cones, orange_cones, optimized_path)
    
    return optimized_path

def save_path_to_csv(optimized_path, filename='optimized_waypoints.csv'):
    with open(filename, 'w') as f:
        f.write('x,y,type\n')
        for x, y in optimized_path:
            f.write(f'{x},{y},Waypoint\n')

if __name__ == "__main__":
    csv_file_path = '/home/deepam/FMDV/FM/Controls/src/fmdv_code/fmdv_code/WaypointsFromPathPlanning.csv'
    optimized_path = main(csv_file_path)
    if optimized_path:
        save_path_to_csv(optimized_path)
