import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

class WasteProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert the web frame to a format OpenCV/YOLO understands
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                # Logic for waste classification
                recyclable = ["bottle", "cup", "can"]
                color = (0, 255, 0) if label in recyclable else (0, 0, 255)

                # Draw the box and label on the frame
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Send the processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("♻️ AI Waste Classifier")
st.write("Grant camera access below to start detecting.")

webrtc_streamer(
    key="waste-detection",
    video_processor_factory=WasteProcessor,
    rtc_configuration={ # This helps the video pass through firewalls
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
