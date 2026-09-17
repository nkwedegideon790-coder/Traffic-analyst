import cv2
import streamlit as st
import sqlite3
import supervision as sv
from datetime import datetime
import time
from ultralytics import YOLO
import tempfile

# ------------------ CONFIG & DB INITIALIZATION ------------------ #
with sqlite3.connect('traffic.db') as conn:
    cursor = conn.cursor()
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

# Active thread connection for database insertions
conn = sqlite3.connect('traffic.db', check_same_thread=False)
cursor = conn.cursor()

# ------------------ LOGGING & FETCH FUNCTIONS ------------------ #
vehicles_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def log_to_db(status, avg_count, current_counts):
    Time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO traffic_data (time, traffic_density, traffic_count, car_count, motorcycle_count, bus_count, truck_count) VALUES (?, ?, ?, ?, ?, ?, ?)", 
        (Time, status, int(avg_count), current_counts['car'], current_counts['motorcycle'], current_counts['bus'], current_counts['truck'])
    )
    conn.commit()

def get_recent_logs(limit: int = 8):
    try:
        read_conn = sqlite3.connect("traffic.db")
        read_cursor = read_conn.cursor()
        read_cursor.execute(
            "SELECT id, time, traffic_density, traffic_count, car_count, motorcycle_count, bus_count, truck_count FROM traffic_data ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = read_cursor.fetchall()
        read_conn.close()
        return rows
    except sqlite3.Error:
        return []

# --------------- MACHINE LEARNING TOOLS INITIALIZATION ------------------ #
model = YOLO('yolov8n.pt')  
box_annotator = sv.BoxAnnotator()
history = [] 
Log_interval = 5  
tracker = sv.ByteTrack()

# ------------- STREAMLIT UI LAYOUT -----------------------------
st.set_page_config(page_title="Traffic Analysis Dashboard", page_icon=":car:", layout="wide")
st.title('Traffic Analysis Dashboard')

uploaded_file = st.file_uploader('Upload video', type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("Video Feed")
        frame_placeholder = st.empty()
    with right_col:
        st.subheader('Recent Logs')
        table_placeholder = st.empty()

    # ⭐ FIX 1: Initialize states outside of loop context so they aren't destroyed every frame
    frame_count = 0
    skip_frame = 5  # Adjusted frame skipping to boost runtime speed
    last_log_time = time.time()
    annotater_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        
        # Fresh count mapping structure for isolated frame visibility metrics
        current_vehicles_count = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

        if frame_count % skip_frame == 0:
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections) # Integrate ByteTrack properly
            
            # Count elements present specifically inside this singular processed frame
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    if class_id in vehicles_classes:
                        vehicle_type = vehicles_classes[class_id]
                        current_vehicles_count[vehicle_type] += 1
            
            traffic_count = sum(current_vehicles_count.values())
            
            def classify(count):
                if count < 5:
                    return "Low"
                elif count < 15:
                    return "Medium"
                else:
                    return "High"          
            
            history.append(traffic_count)
            if len(history) > 10:
                history.pop(0)
            
            avg_count = sum(history) // len(history)
            status = classify(avg_count)

            # ⭐ FIX 2: Fixed logic condition matching system timestamps correctly
            if current_time - last_log_time >= Log_interval:
                log_to_db(status, avg_count, current_vehicles_count)
                last_log_time = current_time

                # ⭐ FIX 3: Fetch fresh database rows inside loop execution dynamically
                live_rows = get_recent_logs(8)
                if live_rows:
                    table_data = [
                        {
                            "Time": row[1], "Density": row[2], "Avg Count": row[3],
                            "Car": row[4], "Motorcycle": row[5], "Bus": row[6], "Truck": row[7]
                        } for row in live_rows
                    ]
                    table_placeholder.table(table_data)

            # Modernized Supervision annotator binding call
            annotater_frame = frame.copy()
            if len(detections) > 0:
                annotater_frame = box_annotator.annotate(scene=annotater_frame, detections=detections)
                cv2.putText(annotater_frame, f'Density: {status} (Avg: {avg_count})', (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            if annotater_frame is None:
                annotater_frame = frame

        try:
            frame_rgb = cv2.cvtColor(annotater_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        except Exception:
            pass

    cap.release()
    st.success('Done processing video.')

    # Final visual confirmation summary query sequence
    final_rows = get_recent_logs(8)
    if final_rows:
        table_data = [
            {
                "Time": row[1], "Density": row[2], "Avg Count": row[3],
                "Car": row[4], "Motorcycle": row[5], "Bus": row[6], "Truck": row[7]
            } for row in final_rows
        ]
        table_placeholder.table(table_data)
