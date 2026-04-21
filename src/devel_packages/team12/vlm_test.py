import pyrealsense2 as rs
import numpy as np
from google import genai
import os
from prompts import *

# --- Configuration ---
# Set your API Key (Recommend using an environment variable)
# API_KEY = ""
api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)

def get_realsense_frame():
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # Convert to numpy array
        return np.asanyarray(color_frame.get_data())
        
    finally:
        # Stop streaming
        pipeline.stop()

# --- Main Logic ---
frame = get_realsense_frame()

if frame is not None:
    # Convert numpy array to bytes for Gemini
    # Gemini expects RGB, but OpenCV/RealSense usually uses BGR
    import cv2
    from PIL import Image
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Prompt Gemini
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            OBJECT_IDENTIFICATION,  
            pil_image
        ]
    )
    
    print("Gemini Response:", response.text)
else:
    print("Failed to capture frame.")
