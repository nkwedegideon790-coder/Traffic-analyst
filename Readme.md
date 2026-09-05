# Traffic Analyst

Traffic Analyst is a real-time traffic monitoring project that detects vehicles from a video stream or webcam, classifies traffic density, and stores summary data in SQLite. The backend exposes a live video stream through FastAPI, while the dashboard is built with Streamlit for a simple web interface.

## Features

- Vehicle detection using YOLOv8
- Traffic density classification: Low, Medium, High
- Bounding-box annotations on each video frame
- FastAPI backend for live streaming and database endpoints
- Streamlit dashboard for monitoring recent traffic results
- SQLite storage for traffic logs and counts

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- Supervision
- FastAPI
- Uvicorn
- Streamlit
- SQLite

## Project Structure

- `main.py` - FastAPI backend and video detection logic
- `UI.py` - Streamlit dashboard
- `requirements.txt` - Python packages
- `traffic.db` - SQLite database generated at runtime
- `traffic.mp4` - optional video source used by the app
- `yolov8n.pt` - default YOLO model file

## Requirements

Python 3.9 or newer is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

### 1) Start the backend

From the project folder:

```bash
python main.py
```

This starts the API server on:

- http://localhost:8000/
- http://localhost:8000/video_feed
- http://localhost:8000/check_db

### 2) Start the dashboard

Open a second terminal and run:

```bash
streamlit run UI.py
```

Then open the local Streamlit URL shown in the terminal, usually:

- http://localhost:8501

## API Endpoints

### GET /

Returns the current backend status.

Example response:

```json
{ "status": "Running" }
```

### GET /video_feed

Streams the live annotated camera feed with detected vehicles and traffic density.

### GET /check_db

Returns the latest logged entries from the SQLite database.

## How It Works

- Frames are read from a source such as `traffic.mp4` or a webcam.
- Every 5th frame is processed by the YOLO model..
- The detected vehicle count is averaged over recent frames to reduce noise.
- The traffic density is classified as:
  - Low: fewer than 5 vehicles
  - Medium: 5 to 14 vehicles
  - High: 15 or more vehicles
- The result is stored in the `traffic_data` table in the `traffic.db` database.

## Database

The application creates a SQLite database named `traffic.db` automatically when it starts.

Schema:

```sql
CREATE TABLE IF NOT EXISTS traffic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    traffic_density TEXT,
    traffic_count INTEGER
)
```

## Video Source Configuration

By default, the backend reads from `traffic.mp4`:

```python
cap = cv2.VideoCapture('traffic.mp4')
```

To use a webcam instead, change it to:

```python
cap = cv2.VideoCapture(0)
```

## Troubleshooting

- If the app shows a video capture error, make sure `traffic.mp4` exists in the project folder or switch to webcam mode.
- If YOLO fails to load, ensure `yolov8n.pt` is present in the project folder.
- If a dependency is missing, reinstall packages with:

```bash
pip install -r requirements.txt
```

## Notes

- The backend and UI are designed to work together: the Streamlit dashboard loads the live stream from the FastAPI endpoint.
- The app is intended for educational and experimental use and can be extended for more advanced traffic analytics.

## License

This project is intended for educational and experimental use.
