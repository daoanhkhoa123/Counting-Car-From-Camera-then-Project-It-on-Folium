from flask import Flask, render_template, Markup, Response
import folium
import random
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

app = Flask(__name__)

# Coordinates for Hanoi and Ho Chi Minh City
hanoi = [21.0285, 105.8542]
ho_chi_minh = [10.8231, 106.6297]
colors = ['red', 'blue', 'green', 'purple', 'orange']

@app.route('/')
def index():
    # Choose a random color for the map line
    random_color = random.choice(colors)
    map_html = create_map(random_color)
    return render_template('index.html', map=Markup(map_html))

def create_map(line_color):
    """
    Create a folium map with a colored line between Hanoi and Ho Chi Minh City
    """
    folium_map = folium.Map(location=[16.0, 108.0], zoom_start=6)
    folium.PolyLine([hanoi, ho_chi_minh], color=line_color, weight=5).add_to(folium_map)
    folium.Marker(hanoi, popup="Hà Nội", icon=folium.Icon(color='blue')).add_to(folium_map)
    folium.Marker(ho_chi_minh, popup="TP.HCM", icon=folium.Icon(color='red')).add_to(folium_map)
    return folium_map._repr_html_()


def gen_frames():
    '''
    Car Counting
    '''
    global car_count, tracked_cars
    # Load YOLO model
    model = YOLO('yolov8s.pt')

    # Lines for counting
    line_1_y = 200
    line_2_y = 400
    tolerance = 6

    # Car count (initialized outside the loop)
    car_count = 0
    tracked_cars = {}

    # Load COCO class names
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    tracker = Tracker()
    cap = cv2.VideoCapture('static/traffic_video.mp4') 

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        detections_df = pd.DataFrame(a).astype("float")

        bbox_list = []

        for index, row in detections_df.iterrows():
            bbox_x1 = int(row[0])
            bbox_y1 = int(row[1])
            bbox_x2 = int(row[2])
            bbox_y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                bbox_list.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

        bbox_id = tracker.update(bbox_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            center_x = int(x3 + x4) // 2
            center_y = int(y3 + y4) // 2

            # Check if car crosses the defined lines
            if line_1_y - tolerance < center_y < line_1_y + tolerance:
                tracked_cars[id] = 'line 1'
            elif line_2_y - tolerance < center_y < line_2_y + tolerance:
                tracked_cars[id] = 'line 2'
            # Draw car position and ID
            if id in tracked_cars:
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Update car count
        car_count = len(tracked_cars)

        # Draw counting lines and car count
        cv2.line(frame, (240, line_1_y), (780, line_1_y), (255, 255, 255), 1)
        cv2.line(frame, (100, line_2_y), (920, line_2_y), (255, 255, 255), 1)
        cv2.putText(frame, f"car_count: {car_count}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-port", type=int, default=5000)
    args = parser.parse_args()

    app.run(port= args.port,debug=True)
