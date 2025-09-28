import sys
import os
import csv
import numpy as np

import cv2
from PIL import Image
from ultralytics import YOLO

"""
FSDS Requirements

"Cameras": {
          "cam": {
            "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 785,
                "Height": 785,
                "FOV_Degrees": 90
            }
            ],
            "X": -0.3,
            "Y": -0.16,
            "Z": 0.8,
            "Pitch": 0.0,
            "Roll": 0.0,
            "Yaw": 0
        }
"""
# Add fsds package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds
"""

client = fsds.FSDSClient("172.18.224.1")

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)
"""

class Perception():
    def __init__(self):
        #car data 
        self.WheelBase = 1.580 #m

        # Load YOLO model
        self.yolo_model = YOLO("C:\\Users\\Arhan\\Desktop\\FMDV\\FSDS_fm\\src\\Control\\best.pt")

    def get_cones(self, client):
        """
        Get the cone positions using YOLO detection on camera image with depth information
        """
        # Get both RGB and depth images from simulator
        image_responses = client.simGetImages([
            fsds.ImageRequest(camera_name='cam', image_type=fsds.ImageType.Scene, 
                            pixels_as_float=False, compress=True),
            fsds.ImageRequest(camera_name='cam', image_type=fsds.ImageType.DepthPerspective, 
                            pixels_as_float=True, compress=False)
        ], vehicle_name='FSCar')
        
        # Processing RGB image
        rgb_response = image_responses[0]
        image_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        image_rgb = image_1d.reshape(rgb_response.height, rgb_response.width, 3)
        
        # Processing depth image
        depth_response = image_responses[1]
        depth_1d = np.array(depth_response.image_data_float, dtype=np.float32)
        depth_image = depth_1d.reshape(depth_response.height, depth_response.width)
        
        # Runing YOLO on RGB image
        results = self.yolo_model(image_rgb)
        
        cones = []
        
        # Processing YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.7:
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        depth = depth_image[center_y, center_x]
                        
                        world_x, world_y, world_z = self.pixel_to_world_with_depth(
                            center_x, center_y, depth, rgb_response.width, rgb_response.height)
                        
                        cones.append([world_x, world_y, world_z, class_id, confidence])
        
        return cones
    
    def pixel_to_world_with_depth(self, pixel_x, pixel_y, depth, image_width, image_height):
        """
        Convert pixel coordinates to world coordinates using depth information
        
        Args:
            pixel_x, pixel_y: pixel coordinates
            depth: depth value at that pixel
            image_width, image_height: image dimensions
        """
        # Camera intrinsic parameters (you may need to adjust these based on your camera)
        # FOV is 90 degrees according to your camera settings
        fov_degrees = 90
        fov_radians = np.radians(fov_degrees)
        
        # Calculate focal length from FOV
        focal_length = (image_width / 2) / np.tan(fov_radians / 2)
        
        # Principal point (assuming center of image)
        cx = image_width / 2
        cy = image_height / 2
        
        # Convert to normalized image coordinates
        x_norm = (pixel_x - cx) / focal_length
        y_norm = (pixel_y - cy) / focal_length
        
        # Convert to 3D camera coordinates
        camera_x = x_norm * depth
        camera_y = y_norm * depth
        camera_z = depth
        
        # Transform from camera coordinates to world coordinates
        # This depends on your camera pose relative to the car
        # Based on your camera settings: X: -0.3, Y: -0.16, Z: 0.8
        world_x = camera_z - 0.3  # Camera is 0.3m behind car center
        world_y = -camera_x - 0.16  # Camera is 0.16m left of car center
        world_z = camera_y + 0.8  # Camera is 0.8m above ground
        
        return world_x, world_y, world_z

    def visualize_detections_with_depth(self, image_rgb, depth_image, results):
        """
        Draw bounding boxes on the image with depth information
        """
        annotated_image = image_rgb.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.5:
                        # Calculate center and get depth
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        depth = depth_image[center_y, center_x]
                        
                        # bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # label with depth information
                        label = f'Cone {confidence:.2f} D:{depth:.2f}m'
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # center point
                        cv2.circle(annotated_image, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # Saving both RGB and depth images
        cv2.imwrite('detection_result.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Normalize depth image for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV8U)
        cv2.imwrite('depth_image.jpg', depth_normalized)
        
        return annotated_image
    
    def filter_valid_depth(self, depth, min_depth=0.1, max_depth=100.0):
        """
        Filter out invalid depth values
        """
        if np.isnan(depth) or np.isinf(depth):
            return False
        if depth < min_depth or depth > max_depth:
            return False
        return True