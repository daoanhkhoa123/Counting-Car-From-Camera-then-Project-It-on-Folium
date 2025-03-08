from flask import Flask, render_template, Response, request
from vidgear.gears import CamGear
import cv2

app = Flask(__name__)

youtube_url = "https://www.youtube.com/watch?v=y8AWFf7EAc4"

video_stream = CamGear(
    source=youtube_url, 
    stream_mode=True,
    logging=True
).start()

# Route for index page
@app.route('/', methods=['GET', 'POST'])
def index():
    global video_stream
    if request.method == 'POST':
        new_url = request.form.get('youtube_url')
        if new_url:
            # Stop the current video stream
            video_stream.stop()
            # Update the video stream with the new URL
            video_stream = CamGear(
                source=new_url, 
                stream_mode=True,
                logging=True
            ).start()
    
    return render_template('index.html')

# Generate video frames and stream
def generate_frames():
    while True:
        frame = video_stream.read()
        if frame is None:
            continue

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
