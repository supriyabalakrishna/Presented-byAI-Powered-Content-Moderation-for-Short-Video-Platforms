import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from ultralytics import YOLO
from collections import deque
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Content Moderation", layout="wide")

st.title(" AI-Powered Content Moderation System")
st.write("Real-time detection using YOLO + Temporal Risk Analysis")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.5
TEMPORAL_WINDOW = 10

RESTRICTED_OBJECTS = {
    "knife": 0.9,
    "scissors": 0.6,
    "fire": 0.8,
    "gun": 1.0
}

frame_buffer = deque(maxlen=TEMPORAL_WINDOW)

# =========================
# FUNCTIONS
# =========================
def calculate_risk(objects):
    risk = 0
    for obj in objects:
        if obj in RESTRICTED_OBJECTS:
            risk += RESTRICTED_OBJECTS[obj]
    return risk

def temporal_risk(buffer):
    all_objs = []
    for objs in buffer:
        all_objs.extend(objs)
    if len(buffer) == 0:
        return 0
    return min(calculate_risk(all_objs) / len(buffer), 1.0)

def decision(score):
    if score > 0.7:
        return "HIGH RISK", "🔴"
    elif score > 0.3:
        return "MEDIUM RISK", "🟠"
    else:
        return "SAFE", "🟢"

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Controls")

option = st.sidebar.selectbox(
    "Choose Input",
    ["Upload Video", "Webcam"]
)

# =========================
# MAIN DISPLAY
# =========================
frame_window = st.empty()
status_box = st.empty()
log_table = st.empty()

logs = []

# =========================
# VIDEO UPLOAD MODE
# =========================
if option == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        st.sidebar.success("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model(frame)

            detected = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    detected.append(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,255,0),2)
                    cv2.putText(frame, label, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0),2)

            frame_buffer.append(detected)

            risk_score = temporal_risk(frame_buffer)
            status, emoji = decision(risk_score)

            # Display
            frame_window.image(frame, channels="BGR")

            status_box.markdown(f"""
            ### {emoji} Status: {status}  
            **Risk Score:** {risk_score:.2f}  
            **Detected Objects:** {detected}
            """)

            logs.append({
                "Time": time.strftime("%H:%M:%S"),
                "Status": status,
                "Risk": risk_score,
                "Objects": str(detected)
            })

            log_table.dataframe(pd.DataFrame(logs))

        cap.release()

# =========================
# WEBCAM MODE
# =========================
elif option == "Webcam":
    run = st.sidebar.button("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model(frame)

            detected = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    detected.append(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,255,0),2)
                    cv2.putText(frame, label, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0),2)

            frame_buffer.append(detected)

            risk_score = temporal_risk(frame_buffer)
            status, emoji = decision(risk_score)

            frame_window.image(frame, channels="BGR")

            status_box.markdown(f"""
            ### {emoji} Status: {status}  
            **Risk Score:** {risk_score:.2f}  
            **Detected Objects:** {detected}
            """)

        cap.release()