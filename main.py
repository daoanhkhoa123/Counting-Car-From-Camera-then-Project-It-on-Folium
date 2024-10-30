from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import *
import torch
import threading
import time

app = Flask(__name__) 

# Load the YOLO model
yolo_model = YOLO('yolov8m.pt')

# Stream video
video_stream = CamGear(source='https://www.youtube.com/watch?v=FsL_KQz4gpw', stream_mode=True, logging=True).start()
vehicle_classes = ['truck', 'car', 'bus', 'motorcycle']
# Global variables for vehicle counting
tracked_vehicles_left = {'car': set(), 'truck': set(), 'bus': set(), 'motorcycle': set()}
tracked_vehicles_right = {'car': set(), 'truck': set(), 'bus': set(), 'motorcycle': set()}
vehicle_counts_left = {class_name: 0 for class_name in vehicle_classes}
vehicle_counts_right = {class_name: 0 for class_name in vehicle_classes}

def check_not_existed_vehicle(vehicle_id):
    global vehicle_classes
    for vehicle_class in vehicle_classes:
        if vehicle_id in tracked_vehicles_left[vehicle_class] or vehicle_id in vehicle_counts_right[vehicle_class]:
            return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vehicle_counts')
def get_vehicle_counts():
    return jsonify({
        'left': vehicle_counts_left,
        'right': vehicle_counts_right
    })

def generate_frames():
    global tracked_vehicles_left, tracked_vehicles_right
    # Lines for counting
    line_y_position = 300
    left_line_x_coords = [330, 470]
    right_line_x_coords = [550, 700]
    tolerance = 6

    # Load COCO class names
    with open("coco.txt", "r") as class_file:
        class_list = class_file.read().split("\n")

    vehicle_tracker = Tracker()

    while True:
        frame = video_stream.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = yolo_model.predict(frame)
        detected_boxes = results[0].boxes.data

        bounding_box_list = []
        for row in detected_boxes:
            bbox_x1 = int(row[0])
            bbox_y1 = int(row[1])
            bbox_x2 = int(row[2])
            bbox_y2 = int(row[3])
            detected_class = class_list[int(row[5])]

            for obj_class in vehicle_classes:
                if obj_class in detected_class:
                    bounding_box_list.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, obj_class])

        bbox_ids, object_classes = vehicle_tracker.update(bounding_box_list)

        for bbox in bbox_ids:
            x3, y3, x4, y4, vehicle_id = bbox
            center_x = int(x3 + x4) // 2
            center_y = int(y3 + y4) // 2

            # Check if vehicle crosses the defined lines
            if line_y_position - tolerance < center_y < line_y_position + tolerance:
                if left_line_x_coords[0] <= center_x <= left_line_x_coords[1]:
                    if vehicle_id not in tracked_vehicles_left['car'] and vehicle_id not in tracked_vehicles_left['truck'] and vehicle_id not in tracked_vehicles_left['bus'] and vehicle_id not in tracked_vehicles_left['motorcycle']:
                        tracked_vehicles_left[object_classes].add(vehicle_id)
                elif right_line_x_coords[0] < center_x <= right_line_x_coords[1]: 
                    if vehicle_id not in tracked_vehicles_right['car'] and vehicle_id not in tracked_vehicles_right['truck'] and vehicle_id not in tracked_vehicles_left['bus'] and vehicle_id not in tracked_vehicles_left['motorcycle']:
                        tracked_vehicles_right[object_classes].add(vehicle_id)

            # Draw vehicle position and ID
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(vehicle_id), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Update vehicle count
        for class_name in vehicle_classes:
            vehicle_counts_left[class_name] = len(tracked_vehicles_left[class_name])
            vehicle_counts_right[class_name] = len(tracked_vehicles_right[class_name])

        cv2.line(frame, (left_line_x_coords[0], line_y_position), (left_line_x_coords[1], line_y_position), (255, 255, 255), 1)
        cv2.line(frame, (right_line_x_coords[0], line_y_position), (right_line_x_coords[1], line_y_position), (255, 255, 255), 1)

        y_position = 30  # Starting position for display
        x_position = 0 
        for class_name in vehicle_classes:
            cv2.putText(frame, f"{class_name} on left road: {vehicle_counts_left[class_name]}", (x_position, y_position), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            y_position += 30
            
        y_position = 30
        x_position = 400  # Starting position for right side display
        for class_name in vehicle_classes:
            cv2.putText(frame, f"{class_name} on right road: {vehicle_counts_right[class_name]}", (x_position, y_position), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            y_position += 30

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_value():
    return vehicle_counts_left, vehicle_counts_right

def update_vehicle_counts():
    while True:
        time.sleep(5)  # Wait for 5 seconds
        # Get value
        left_counts, right_counts = get_value()
        print("Vehicle counts left:", left_counts)
        print("Vehicle counts right:", right_counts)

        # Xóa dữ liệu trong tracked_vehicles
        for vehicle_class in vehicle_classes:
            tracked_vehicles_left[vehicle_class].clear()
            tracked_vehicles_right[vehicle_class].clear()

if __name__ == '__main__':
    # Start a thread to update vehicle counts every minute
    threading.Thread(target=update_vehicle_counts, daemon=True).start()
    app.run(debug=True)