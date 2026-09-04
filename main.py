import cv2
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
                        traffic_count INTEGER,
                        car_count INTEGER,
                        motorcycle_count INTEGER,
                        bus_count INTEGER,
                        truck_count INTEGER
                    )''')
    conn.commit()
# ------------------ DATABASE CONNECTION ------------------ #
conn = sqlite3.connect('traffic.db', check_same_thread=False)
cursor = conn.cursor()
# ------------------ LOGGING FUNCTION ------------------ #
vehicles_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
def log_to_db(status, avg_count, vehicles_count):
    Time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Insert the data into the database
    cursor.execute("INSERT INTO traffic_data (time, traffic_density, traffic_count, car_count, motorcycle_count, bus_count, truck_count) VALUES (?, ?, ?, ?, ?, ?, ?)", (Time, status, avg_count, vehicles_count['car'], vehicles_count['motorcycle'], vehicles_count['bus'], vehicles_count['truck']))
    conn.commit()

app = FastAPI()

# ------------------ VIDEO CAPTURE AND MODEL ------------------ #
cap = cv2.VideoCapture('traffic.mp4')  # or 0 for webcam
model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
box_annonator = sv.BoxAnnotator() #// Create a box annotator for drawing bounding boxes
history = [] # get the history of traffic counts for averaging 
Log_interval = 5  # Log every 5 seconds
Tracker = sv.ByteTrack()

# ------------------ ROUTES ------------------ #

@app.get('/')
def stats():
    return {"status": "Running"}

# ------------------ FRAME GENERATOR ------------------ #

def frame_generator():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 340)
    frame_count = 0
    skip_frame = 10
    last_detection = None
    last_log_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frame == 0:
            # YOLO detection
            results = model(frame, classes = [2,3,5,7], conf=0.6, imgsz=416, verbose=False)[0]
            detections =sv.Detections.from_ultralytics(results)
            detections = Tracker.update_with_detections(detections)
            last_detection = detections
            # detections = Tracker.update_with_detections(detections)
            vehicles_count = {
                'car': 0,
                'motorcycle': 0,
                'bus': 0,
                'truck': 0
            }
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in vehicles_classes:
                        vehicle_type = vehicles_classes[class_id]
                        vehicles_count[vehicle_type] += 1
       
            traffic_count = sum(vehicles_count.values())
            # classify density of traffic
            def classify(traffic_count):
                if traffic_count < 5:
                    return "Low"
                elif len(detections) < 15:
                    return "Medium"
                else:
                    return "High"
        
             # Calculate average traffic count over the last 10 frames
            current_time = time.time()
            history.append(traffic_count)
            if len(history)>10:
                history.pop(0)

            # Calculate average traffic count and classify density
            avg_count = sum(history)//len(history)
            status = classify(avg_count)
            if current_time - last_log_time >= Log_interval:
                log_to_db(status,avg_count, vehicles_count)
                last_log_time = current_time
        else:
            detections = last_detection
        # Annotate the frame with traffic density information
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
    frame_count += 1
    cap.release()
# ------------------ VIDEO STREAM ------------------ #

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# ------------------ DATABASE CHECK ------------------ #
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

