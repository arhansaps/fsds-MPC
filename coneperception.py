import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv
import math



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

client = fsds.FSDSClient("127.0.0.1")
client.confirmConnection()
client.enableApiControl(True)

carState = client.getCarState(vehicle_name='FSCar')
carState = carState.kinematics_estimated


tracked_cones = {}
final_cones = {}
cone_id_counter = 0
cone_radius = 1.0
written_cones = {}

model = YOLO("C:\\Users\\Arhan\\Desktop\\FMDV\\FSDS_fm\\src\\Control\\best.pt")

frames_seen_threshold = 2

W, H = 256, 144
cx_pixel_center = W/2
cy_pixel_center = H/2
FOV_deg = 90
fx = fy = (W / 2) / np.tan(np.deg2rad(FOV_deg / 2))
cx, cy = W / 2, H / 2

euler_angles = np.array([0, 0, 0])
rot = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
trans = np.array([-0.3, -0.16, 0.8])

def get_yaw_from_quaternion(q):
    t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    t4 = +1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return np.arctan2(t3, t4)

def local_to_global_fsds(x_local, y_local, carState):
    x_car = carState.position.x_val
    y_car = carState.position.y_val
    yaw = get_yaw_from_quaternion(carState.orientation)
    x_global = x_car + x_local * np.cos(yaw) - y_local * np.sin(yaw)
    y_global = y_car + x_local * np.sin(yaw) + y_local * np.cos(yaw)
    return x_global, y_global

with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'w', newline='') as file:
    pass

with open(os.path.join(os.path.dirname(__file__),'finalcones.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cone_id', 'x', 'y', 'color'])

def append_values_to_csv(row):
    with open(os.path.join(os.path.dirname(__file__),'carpos.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def append_cones_to_csv():
    global written_cones
    with open(os.path.join(os.path.dirname(__file__),'finalcones.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        for cone_id, data in final_cones.items():
            pos = data['pos']
            color = data['color']
            if cone_id not in written_cones or not np.allclose(pos, written_cones[cone_id]['pos'], rtol=1e-5) or color != written_cones[cone_id]['color']:
                writer.writerow([cone_id, pos[0], pos[1], color])
                written_cones[cone_id] = {'pos': pos, 'color': color}

time_period = 1/30

class ConePerception():
    def perform_cone_detection(car):
        global cone_id_counter, tracked_cones, final_cones

        [image] = client.simGetImages([
            fsds.ImageRequest(camera_name='cam2', image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=True)
        ], vehicle_name='FSCar')

        img_np = np.frombuffer(image.image_data_uint8, dtype=np.uint8)
        original_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        H_orig, W_orig = original_img.shape[:2]

        [imageDepth] = client.simGetImages([
            fsds.ImageRequest(camera_name='cam1', image_type=fsds.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
    ], vehicle_name='FSCar')

        depth_array = np.array(imageDepth.image_data_float, dtype=np.float32)
        depth_image = depth_array.reshape((imageDepth.height, imageDepth.width))

        resized_img = cv2.resize(original_img, (imageDepth.width, imageDepth.height))
        rgb_image = resized_img.copy()

        results = model(resized_img)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        names = model.names

        centers = []
        car_frame_coords = []
        detections = []
        left_count = 0
        right_count = 0
        max_per_side = 3

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                depth = depth_image[cy, cx]
                if depth is not None:
                    x_cam = (cx - cx) * depth / fx
                if x_cam < 0 and left_count < max_per_side:
                    centers.append((cx, cy))
                    detections.append({'center': (cx, cy), 'depth': depth, 'conf': conf, 'color': names[cls]})
                    left_count += 1
                elif x_cam >= 0 and right_count < max_per_side:
                    centers.append((cx, cy))
                    detections.append({'center': (cx, cy), 'depth': depth, 'conf': conf, 'color': names[cls]})
                    right_count += 1
                if left_count >= max_per_side and right_count >= max_per_side:
                    break

        for det in detections:
            cx, cy, depth = det['center'][0], det['center'][1], det['depth']
            x_cam = (cx - cx_pixel_center) * depth / fx
            y_cam = (cy - cy_pixel_center) * depth / fy
            z_cam = depth
            car_frame_coords.append((x_cam, y_cam, z_cam, det['conf'], det['color']))

        
        global_coords = []
        for x_rel, _, z_rel, conf, color in car_frame_coords:
            x_global, y_global = local_to_global_fsds(x_rel, z_rel, carState)
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
                tracked_cones[best_id]['frames_seen'] += 1 #new
                positions = [obs['pos'] for obs in tracked_cones[best_id]['observations']]
                weights = [obs['conf'] for obs in tracked_cones[best_id]['observations']]
                tracked_cones[best_id]['center'] = np.average(positions, axis=0, weights=weights)
                updated_ids.add(best_id)
            else:
                cone_id_counter += 1
                tracked_cones[cone_id_counter] = {
                'observations': [{'pos': new_pos, 'conf': new_conf, 'color': new_color}],
                'center': new_pos,
                'frames_seen': 1 #new 
            }
                updated_ids.add(cone_id_counter)
        
        to_delete = []

        for cone_id, data in list(tracked_cones.items()):
            print("checkpt 1")
            # 2 cones get added without the 2nd condition idk why
            #if len(data['observations']) > 0: and cone_id not in updated_ids:
            #trying new condition
            if data['frames_seen'] >= frames_seen_threshold:
                positions = [obs['pos'] for obs in data['observations']]
                weights = [obs['conf'] for obs in data['observations']]
                colors = [obs['color'] for obs in data['observations']]
                print("checkpt 2")

                #wted average

                final_pos = np.average(positions,axis=0,weights=weights)

                color_counts = {}
                for color in colors:
                    color_counts[color] = color_counts.get(color, 0) + 1
                final_color = max(color_counts, key=color_counts.get)
                
                final_cones[cone_id] = {'pos': final_pos, 'color': final_color}

                to_delete.append(cone_id)
        
        for cone_id in to_delete:
            del tracked_cones[cone_id]

        if final_cones:
            print("\nFinal Cones (weighted averages):")
            for cone_id, data in final_cones.items():
                print(f"Cone {cone_id}: Pos={data['pos'].round(4)}, Color={data['color']}")

        append_cones_to_csv()


