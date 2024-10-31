from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import *
import torch
import threading
import time
import folium
from numpy.random import choice
import time
import random 
from traffic_pred.traffic_prediction import Traiffic_Classifier
app = Flask(__name__) 

# Load the YOLO model
yolo_model = YOLO('yolov8m.pt')
traffic_model = Traiffic_Classifier()

# Stream video
video_stream = CamGear(source='https://www.youtube.com/watch?v=wqctLW0Hb_0', stream_mode=True, logging=True).start()  # not stream
# video_stream = CamGear(source='https://www.youtube.com/watch?v=FsL_KQz4gpw', stream_mode=True, logging=True).start()
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
    return render_template('layout.html')

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
    left_line_x_coords = [100, 470]
    right_line_x_coords = [500, 900]
    # left_line_x_coords = [330, 470]
    # right_line_x_coords = [550, 700]
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


def update_vehicle_counts():
    while True:
        time.sleep(50)  # Wait for 5 seconds
        
        print("Vehicle counts left:", vehicle_counts_left)
        print("Vehicle counts right:", vehicle_counts_right)

        # Xóa dữ liệu trong tracked_vehicles
        for vehicle_class in vehicle_classes:
            tracked_vehicles_left[vehicle_class].clear()
            tracked_vehicles_right[vehicle_class].clear()
            

def get_volumn():
    global traffic_model
    total = sum(vehicle_counts_left.values())
    predict_input = np.array([vehicle_counts_left['car'],
                              vehicle_counts_left['motorcycle'],
                              vehicle_counts_left['bus'],
                              vehicle_counts_left['truck'],
                              total]).reshape(1, -1)*15
    print(predict_input)
    pred_volumn = traffic_model.predict_text(predict_input)[0]
    print("pred_volumn:", pred_volumn)
    return [total, pred_volumn]
@app.route("/generate_map")
def generate_map():
    gen_map()  # This will update the map with new data
    return {"status": "Map updated"}

def gen_map():
    market_street_coords = [
        [37.774929, -122.419416],
        [37.793731, -122.394242]
    ]

    m = folium.Map(location=[37.7749, -122.4194], zoom_start=13)
    traffic_data = get_volumn()
    color_map = {'low': 'green', 'normal': 'yellow', 'heavy': 'orange', 'high': 'red'}
    print("traffic_data", traffic_data)
    color = color_map[traffic_data[1]]
    print("color", color)
    folium.PolyLine(market_street_coords, color=color, weight=8, opacity=0.6).add_to(m)
    print("Folium ran ")
    m.save('static/map.html')

if __name__ == '__main__':
    threading.Thread(target=update_vehicle_counts, daemon=True).start()
    gen_map()
    app.run(debug=True)