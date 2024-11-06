from collections import deque
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, Response, jsonify
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import *
from traffic_pred.traffic_prediction import Traffic_Classifier
import time
import matplotlib.dates as mdates
import matplotlib

matplotlib.use('Agg')


app = Flask(__name__)

# Load the YOLO model and traffic classifier
yolo_model = YOLO('yolov8s.pt')
traffic_model = Traffic_Classifier()

video_stream = CamGear(source='https://www.youtube.com/watch?v=wqctLW0Hb_0',
                       stream_mode=True, logging=True).start()  # not stream -> test detect
# # Initialize video stream
# video_stream = CamGear(source='https://www.youtube.com/watch?v=FsL_KQz4gpw',
                    #    stream_mode=True, logging=True).start() # stream x

# video_stream = CamGear(source='https://www.youtube.com/watch?v=Y1jTEyb3wiI',
#                        stream_mode=True, logging=True).start() # not stream -> test line 

vehicle_classes = ['car', 'truck',  'bus', 'motorcycle']
color_map = {'low': 'green', 'normal': 'yellow',
             'heavy': 'orange', 'high': 'red'}

# Initialize vehicle counts as a dictionary
traffic_pie = np.zeros(4, dtype=int)
vehicle_counts = np.zeros(4, dtype=int)

traffic_data = [0, "low"]
utc_plus_9 = timezone(timedelta(hours=9))
queue = deque(maxlen=10)

# Load class names from coco.txt only once
with open("coco.txt", "r") as class_file:
    class_list = class_file.read().split("\n")


@app.route('/')
def index():
    return render_template('index.html')



def generate_frames():
    global vehicle_counts, traffic_data, traffic_pie
    line_x_coords1 = 300
    line_x_coords2 = 850
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
            detected_class = class_list[int(row[5])] # float 4.312

            for idx, obj_class in enumerate(vehicle_classes):
                # if obj_class in detected_class and (line_x_coords1 < bbox_x1 < line_x_coords2):
                if obj_class in detected_class:
                    bounding_box_list.append(
                        [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
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

         # Define rectangle parameters
        x, y, w, h = 0, 0, 315, 170  # Position and size of the rectangle
        overlay = frame.copy()  # Copy the frame to an overlay

        # Draw a filled black rectangle on the overlay
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

        # Blend the overlay with the frame to create transparency
        alpha = 0.7  # Transparency factor (0: transparent, 1: opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display vehicle counts on frame
        y_position = 60
        for idx, class_name in enumerate(vehicle_classes):
            cv2.putText(frame, f"{class_name} on road: {vehicle_counts[idx]}",
                        (0, y_position), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            y_position += 30

        cv2.line(frame, (line_x_coords1, 0),
                 (line_x_coords1, 900), (255, 255, 255), 1)
        
        # cv2.line(frame, (line_x_coords2, 0),
        #          (line_x_coords2, 900), (255, 255, 255), 1)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame) # image encode 
        frame = buffer.tobytes() # covert pic to bytes 
        traffic_data = get_volume()
        traffic_pie = vehicle_counts.copy()
        traffic_pie[0] += 1

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_volume():
    global traffic_model, vehicle_counts
    print(vehicle_counts)

    # Ensure predict_input has the correct shape for the model
    predict_input = vehicle_counts.reshape(1, -1)
    # Tính tổng cho cột cuối cùng
    total_count = np.sum(predict_input[0, :-1])
    predict_input = np.append(predict_input, total_count).reshape(1, -1)

    print(predict_input)

    # Predict traffic volume
    pred_volume = traffic_model.predict_text(predict_input.reshape(1, -1))[0]
    print(pred_volume)

    return [total_count, pred_volume]


@app.route("/generate_")
def generate_map():
    market_street_coords, color = gen_map()
    return jsonify({'coords': market_street_coords, 'color': color})


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

    color = color_map[traffic_data[1]]
    return market_street_coords, color


@app.route('/generate_pie_chart')
def generate_pie_chart():
    global traffic_pie, queue
    # Sample data for the pie chart
    labels = ['Car', 'Truck', 'Bus', 'Motorcycle']
    sizes = list(traffic_pie)
    fig, ax = plt.subplots()

    non_zero_labels = [labels[i] for i in range(len(labels)) if sizes[i] != 0] 
    sizes = [s for s in sizes if s != 0]
    ax.pie(sizes, labels=non_zero_labels, autopct='%1.1f%%')

    # Save the plot to an in-memory file
    plt.legend(loc='upper right')
    plt.savefig('static/pie_chart.png')

    plt.close(fig)  # Close the figure to release memory
    print("pied")

    fig, ax = plt.subplots()
    ax.set_title("Real-time Time Series Plot")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Vehicle counts")
    # ax.set_ylim(0, 25)  # Set the y-axis range from 0 to 100
    # Initialize an empty line for the plot
    line, = ax.plot([], [], 'b-', marker='o')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # Get the current time with the UTC+9 timezone
    current_time_utc_plus_9 = datetime.now(utc_plus_9)
    print('curr time: ', current_time_utc_plus_9, type(current_time_utc_plus_9))
    
    current_time_utc_plus_9 = current_time_utc_plus_9 + timedelta(hours= +9)
    # Automatically removes oldest if full
    queue.append([current_time_utc_plus_9, np.sum(traffic_pie)])

    # Extract data from the queue for plotting
    times, values = zip(*queue)
    print("times:", times)
    # Update the plot's x and y data
    line.set_data(times, values)
    ax.set_xticks(times)  # Set x-axis ticks to the timestamp values
    # Set y-axis ticks from 0 to 100 with steps of 10
    ax.set_yticks(range(0, 30, 10))
    ax.relim()         # Adjust plot limits
    ax.autoscale_view()  # Rescale view to fit new data

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('static/time_series.png')

    plt.close(fig)

    # Return the in-memory file as a response
    return Response(status=200)


if __name__ == '__main__':
    gen_map()
    app.run(debug=True)
