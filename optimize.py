from collections import deque
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, Response, jsonify, url_for, send_file
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import *
import folium
from traffic_pred.traffic_prediction import Traiffic_Classifier
import time
import matplotlib.dates as mdates
import matplotlib

matplotlib.use('Agg')


app = Flask(__name__)

# Load the YOLO model and traffic classifier
yolo_model = YOLO('yolov8m.pt')
traffic_model = Traiffic_Classifier()

video_stream = CamGear(source='https://www.youtube.com/watch?v=wqctLW0Hb_0',
                       stream_mode=True, logging=True).start()  # not stream
# Initialize video stream
# video_stream = CamGear(source='https://www.youtube.com/watch?v=FsL_KQz4gpw',
#                        stream_mode=True, logging=True).start()

vehicle_classes = ['car','truck',  'bus', 'motorcycle']
color_map = {'low': 'green', 'normal': 'yellow', 'heavy': 'orange', 'high': 'red'}

# Initialize vehicle counts as a dictionary
traffic_pie = np.zeros(4, dtype=int)
vehicle_counts = np.zeros(len(vehicle_classes), dtype=int)

traffic_data = [0, "low"]
utc_plus_9 = timezone(timedelta(hours=9))
queue = deque(maxlen=10)

# Load class names from coco.txt only once
with open("coco.txt", "r") as class_file:
    class_list = class_file.read().split("\n")


@app.route('/')
def index():
    return render_template('layout.html')


@app.route('/vehicle_counts')
def get_vehicle_counts():
    return jsonify({
        'counts': dict(zip(vehicle_classes, vehicle_counts.tolist())),
    })


def generate_frames():
    global vehicle_counts, traffic_data, traffic_pie
    line_x_coords = 300
    vehicle_tracker = Tracker()

    frame_counter = 0
    fps = 0
    prev_time = time.time()
    
    while True:
        vehicle_counts[:] = 0  # Reset vehicle counts for each frame
        frame = video_stream.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1020, 500))
        
        # Predict on every frame for accurate tracking
        results = yolo_model.predict(frame)
        detected_boxes = results[0].boxes.data
        bounding_box_list = []

        for row in detected_boxes:
            bbox_x1 = int(row[0])
            bbox_y1 = int(row[1])
            bbox_x2 = int(row[2])
            bbox_y2 = int(row[3])
            detected_class = class_list[int(row[5])]

            for idx, obj_class in enumerate(vehicle_classes):
                # if obj_class in detected_class and bbox_x1 > line_x_coords:
                if obj_class in detected_class :
                    bounding_box_list.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])
                    vehicle_counts[idx] += 1

        bbox_ids = vehicle_tracker.update(bounding_box_list)
        for bbox in bbox_ids:
            x3, y3, x4, y4, vehicle_id = bbox
            center_x = int(x3 + x4) // 2
            center_y = int(y3 + y4) // 2
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(vehicle_id), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
        frame_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_counter
            frame_counter = 0
            prev_time = current_time

        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display vehicle counts on frame
        y_position = 60
        for idx, class_name in enumerate(vehicle_classes):
            cv2.putText(frame, f"{class_name} on road: {vehicle_counts[idx]}",
                        (0, y_position), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            y_position += 30

        cv2.line(frame, (line_x_coords, 0), (line_x_coords, 900), (255, 255, 255), 1)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        traffic_data = get_volume()
        traffic_pie = vehicle_counts.copy()
        traffic_pie[0]+=1
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_volume():
    global traffic_model, vehicle_counts
    print(vehicle_counts)
    
    # Ensure predict_input has the correct shape for the model
    predict_input = (vehicle_counts * 15).reshape(1, -1) + np.random.randint(0, 10, size=vehicle_counts.size)
    
    # Tính tổng cho cột cuối cùng
    total_count = np.sum(predict_input[0, :-1]) 
    predict_input = np.append(predict_input, total_count).reshape(1, -1) 
    
    print(predict_input)
    
    # Predict traffic volume
    pred_volume = traffic_model.predict_text(predict_input.reshape(1, -1))[0]
    print(pred_volume)
    
    return [total_count, pred_volume]



@app.route("/generate_map")
def generate_map():
    gen_map()
    return {"status": "Map updated"}


def gen_map():
    global traffic_data, color_map, traffic_pie
    market_street_coords = [
        [35.67622529435731, 139.31493925615086],
        [35.67341420095618, 139.3094525105975],
        [35.66839384886371, 139.30260751315072],
        [35.666336807784944, 139.29782245226014],
        [35.665883554324274, 139.29606292314352],
        [35.6649073073867, 139.29140660826593],
        [35.66203079604145, 139.28614947860407],
    ]

    m = folium.Map(location=[35.668215952240345, 139.30232459692803], zoom_start=14)
    color = color_map[traffic_data[1]]
    folium.PolyLine(market_street_coords, color=color, weight=8, opacity=0.6).add_to(m)
    m.save('static/map.html')

@app.route('/generate_pie_chart')
def generate_pie_chart():
    global traffic_pie, queue
    # Sample data for the pie chart
    labels = ['Car', 'Truck', 'Bus', 'Motorcycle']
    time_pie = time.strftime("%Y-%m-%dT%H:%M:%S")
    sizes =  list(traffic_pie)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    
    # Save the plot to an in-memory file
    plt.savefig('static/pie_chart.png')
    
    plt.close(fig)  # Close the figure to release memory
    print("pied")

    # Define a timezone offset for UTC+9

    # Initialize a deque with a maximum length of 10 (adjustable)

    # Set up the plot




    fig, ax = plt.subplots()
    ax.set_title("Real-time Time Series Plot")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Random Value")
    ax.set_ylim(0, 25)  # Set the y-axis range from 0 to 100
    line, = ax.plot([], [], 'b-', marker='o')  # Initialize an empty line for the plot
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # Get the current time with the UTC+9 timezone
    current_time_utc_plus_9 = datetime.now(utc_plus_9)
    print('curr time: ',current_time_utc_plus_9, type(current_time_utc_plus_9))

    queue.append([current_time_utc_plus_9,np.sum(traffic_pie)])  # Automatically removes oldest if full
    
    # Extract data from the queue for plotting
    times, values = zip(*queue)
    
    # Update the plot's x and y data
    line.set_data(times, values)
    ax.set_xticks(times)  # Set x-axis ticks to the timestamp values
    ax.set_yticks(range(0, 30, 10))  # Set y-axis ticks from 0 to 100 with steps of 10
    ax.relim()         # Adjust plot limits
    ax.autoscale_view() # Rescale view to fit new data
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('static/time_series.png')
    
    plt.close(fig)

    # Return the in-memory file as a response
    return 

if __name__ == '__main__':
    gen_map()
    app.run(debug=True)
