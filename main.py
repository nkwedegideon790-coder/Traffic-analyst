import cv2
from polars import var
from ultralytics import YOLO
import supervision as sv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime
import uvicorn
import sqlite3
import time

# ------------------ CONFIG ------------------ #
with sqlite3.connect('traffic.db') as conn:
    cursor = conn.cursor()
    # Create the traffic_data table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS traffic_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        time TEXT,
                        traffic_density TEXT,
                        traffic_count INTEGER
                    )''')
    conn.commit()
conn = sqlite3.connect('traffic.db', check_same_thread=False)
cursor = conn.cursor()
def log_to_db(status, avg_count):
    Time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Insert the data into the database
    cursor.execute("INSERT INTO traffic_data (time, traffic_density, traffic_count) VALUES (?, ?, ?)", (Time, status, avg_count))
    conn.commit()

app = FastAPI()

cap = cv2.VideoCapture('traffic.mp4')  # or 0 for webcam
model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
box_annonator = sv.BoxAnnotator()
line_annonator = sv.LineZoneAnnotator()
label_annonator = sv.LabelAnnotator()
history = []



Log_interval = 5  # Log every 5 seconds
# ------------------ ROUTES ------------------ #

@app.get('/')
def stats():
    return {"status": "Running"}

# ------------------ FRAME GENERATOR ------------------ #

def frame_generator():

    last_log_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        # YOLO detection
        results = model(frame, classes = [2,3,5,7], conf=0.6, imgsz=416, verbose=False)[0]
        detections =sv.Detections.from_ultralytics(results)
        # detections = Tracker.update_with_detections(detections)
        traffic_count = len(detections)

        # classify density of traffic
        def classify(traffic_count):
            if traffic_count < 5:
                return "Low"
            elif len(detections) < 15:
                return "Medium"
            else:
                return "High"
        
        current_time = time.time()
        history.append(traffic_count)
        if len(history)>10:
            history.pop(0)


        avg_count = sum(history)//len(history)
        status = classify(avg_count)
        if current_time - last_log_time >= Log_interval:
            log_to_db(status,avg_count)
            last_log_time = current_time

        annonater_frame = box_annonator.annotate(scene=frame, detections=detections)
        cv2.putText(annonater_frame, f'Traffic Density: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', annonater_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# ------------------ VIDEO STREAM ------------------ #

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get('/check_db')
def check_db():
    conn = sqlite3.connect('traffic.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM traffic_data ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    
    conn.close()
    return {"data": rows}
# ------------------ RUN ------------------ #

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

