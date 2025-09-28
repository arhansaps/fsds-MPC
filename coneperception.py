import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import sys
import os
import numpy as np
import time
import csv
import math


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

# Hook up to the simulator
client = fsds.FSDSClient("127.0.0.1")
client.confirmConnection()
client.enableApiControl(True)

tracked_cones = {}  # Store cone IDs with their obs and center
final_cones = {}    # Final cone spots
cone_id_counter = 0
cone_radius = 1.0

# Load that YOLO model
model = YOLO("C:\\Users\\Arhan\\Desktop\\FMDV\\FSDS_fm\\src\\Control\\best.pt")

# Camera setup
W, H = 256, 144
FOV_deg = 90
fx = fy = (W / 2) / np.tan(np.deg2rad(FOV_deg / 2))
cx, cy = W / 2, H / 2

# Camera position stuff
euler_angles = np.array([0, -5, 0])
rot = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
trans = np.array([-0.3, 0, 0.8])

def get_yaw_from_quaternion(q):
    t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    t4 = +1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return np.arctan2(t3, t4)

def local_to_global_fsds(x_local, z_local, car_state):
    x_car = car_state.position.x_val - 2
    y_car = car_state.position.y_val
    yaw = get_yaw_from_quaternion(car_state.orientation)
    x_global = x_car + x_local * np.cos(yaw) - z_local * np.sin(yaw)
    y_global = y_car + x_local * np.sin(yaw) + z_local * np.cos(yaw)
    return x_global, -y_global

# File setup
with open('carpos.csv', 'w', newline='') as file:
    pass
with open('cones.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['tag', 'x', 'y', 'direction', 'x_variance', 'y_variance', 'xy_covariance'])

def append_values_to_csv(row):
    with open('carpos.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def append_cones_to_csv():
    with open('cones.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['tag', 'x', 'y', 'direction', 'x_variance', 'y_variance', 'xy_covariance'])
        if len(final_cones) > 90:
            sorted_cones = dict(sorted(final_cones.items())[:90])
        else:
            sorted_cones = final_cones
        for cone_id, data in sorted_cones.items():
            pos = data['pos']
            color = data['color']
            writer.writerow([color, pos[0], pos[1], 0.0, 0.01, 0.01, 0.0])

class ConeDetection:
    def perform_cone_detection(car):
        global cone_id_counter, tracked_cones, final_cones

        [image] = client.simGetImages([fsds.ImageRequest(camera_name='cam1', image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=True)], vehicle_name='FSCar')
        img_np = np.frombuffer(image.image_data_uint8, dtype=np.uint8)
        original_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        [imageDepth] = client.simGetImages([fsds.ImageRequest(camera_name='cam1', image_type=fsds.ImageType.DepthPerspective, pixels_as_float=True, compress=False)], vehicle_name='FSCar')
        depth_array = np.array(imageDepth.image_data_float, dtype=np.float32)
        depth_image = depth_array.reshape((H, W))

        resized_img = cv2.resize(original_img, (W, H))
        results = model(resized_img)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        names = model.names

        car_frame_coords = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if 0 <= cx < W and 0 <= cy < H:
                depth = depth_image[cy, cx]
                if 0.5 <= depth <= 50:
                    x_cam = (cx - 128) * depth / fx
                    y_cam = (cy - 72) * depth / fy
                    z_cam = depth
                    point_cam = np.array([x_cam, y_cam, z_cam])
                    local_pos = np.dot(rot, point_cam) + trans
                    x_local, y_local, z_local = local_pos
                    car_frame_coords.append((x_local, y_local, z_local, conf, names[cls]))

        global_coords = []
        for x_rel, y_rel, z_rel, conf, color in car_frame_coords:
            x_global, y_global = local_to_global_fsds(x_rel, z_rel, car.carState)
            global_coords.append({'pos': np.array([x_global, y_global]), 'conf': conf, 'color': color})

        updated_ids = set()
        for detection in global_coords:
            new_pos = detection['pos']
            new_conf = detection['conf']
            new_color = detection['color']
            best_id = None
            min_dist = float('inf')
            for cone_id, data in tracked_cones.items():
                center = data['center']
                dist = np.linalg.norm(new_pos - center)
                if dist < cone_radius and dist < min_dist:
                    min_dist = dist
                    best_id = cone_id
            if best_id is not None:
                tracked_cones[best_id]['observations'].append({'pos': new_pos, 'conf': new_conf, 'color': new_color})
                positions = [obs['pos'] for obs in tracked_cones[best_id]['observations']]
                weights = [obs['conf'] for obs in tracked_cones[best_id]['observations']]
                tracked_cones[best_id]['center'] = np.average(positions, axis=0, weights=weights)
                updated_ids.add(best_id)
            else:
                cone_id_counter += 1
                tracked_cones[cone_id_counter] = {'observations': [{'pos': new_pos, 'conf': new_conf, 'color': new_color}], 'center': new_pos}
                updated_ids.add(cone_id_counter)

        for cone_id, data in list(tracked_cones.items()):
            if cone_id not in updated_ids and len(data['observations']) > 0:
                positions = [obs['pos'] for obs in data['observations']]
                weights = [obs['conf'] for obs in data['observations']]
                colors = [obs['color'] for obs in data['observations']]
                color_counts = {}
                for color in colors:
                    color_counts[color] = color_counts.get(color, 0) + 1
                final_color = max(color_counts, key=color_counts.get)
                final_pos = np.average(positions, axis=0, weights=weights)
                final_cones[cone_id] = {'pos': final_pos, 'color': final_color}
                del tracked_cones[cone_id]

        if final_cones:
            append_cones_to_csv()




