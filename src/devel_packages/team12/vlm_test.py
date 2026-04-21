import os
import time

import pyrealsense2 as rs
from google import genai
import json
import re
import numpy as np

rack_analyzer_prompt = """
        Analyze this image of a 3x3 storage rack. The grid positions are: top-left, top-center, top-right, mid-left, mid-center, mid-right, low-left, low-center, low-right.

        For each of the 9 locations, provide:
        1. "status": 'empty' or 'occupied'.
        2. "category": The primary industry or use-case category (e.g., 'tools', 'stationery', 'kitchenware'). If empty, use 'none'.
        3. "theme": A short, descriptive phrase identifying the common semantic theme of all objects in that specific slot (e.g., 'office supplies for writing', 'heavy-duty fasteners'). If empty, use 'none'.

        Output ONLY valid JSON following this structure:
        {
          "rack_audit": {
            "top-left": {"status": "...", "category": "...", "theme": "..."},
            "top-center": {"status": "...", "category": "...", "theme": "..."},
            "top-right": {"status": "...", "category": "...", "theme": "..."},
            "mid-left": {"status": "...", "category": "...", "theme": "..."},
            "mid-center": {"status": "...", "category": "...", "theme": "..."},
            "mid-right": {"status": "...", "category": "...", "theme": "..."},
            "low-left": {"status": "...", "category": "...", "theme": "..."},
            "low-center": {"status": "...", "category": "...", "theme": "..."},
            "low-right": {"status": "...", "category": "...", "theme": "..."}
          }
        }
        """


semantic_classifier_prompt = """
You are a robotic librarian. Below is the current semantic state of the rack and an image of a new object that needs to be placed.

Current Rack Audit:
{{RACK_AUDIT_JSON_FROM_PASS_1}}

Task:
1. Analyze the semantic relationship between the new object and the existing categories/themes in the rack.
2. Select the optimal location for the new object to maintain logical organization. 
3. If the rack is empty, pick a logical starting position (e.g., 'top-left').
4. If the rack is partially full, prioritize empty slots that align with the new object's semantic theme.

Output ONLY valid JSON:
{
  "chosen_location": "...",
  "reasoning": "Explain why this location is semantically aligned with the new object's category and theme.",
}
"""


class vlm_autonomy:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-3-flash-preview"
        genai.configure(api_key=self.api_key)

        self.rack_analyzer_prompt = rack_analyzer_prompt
        self.semantic_classifier_prompt = semantic_classifier_prompt
        self.data = {}


    def get_response(self, image, prompt):
        response = self.model.models.generate_content(
            model=self.model_id,
            contents=[
                prompt,
                image
            ]
        )

        return response

    def analyze_rack(self, rack_img):
        """
        Takes the current Screen shot of the rack and outputs a structured JSON
        containing categories of all the 9 segments of the rack.

        Invokes gemini call to obtain JSON

        :param rack_img:
        :return:
        """
        response = self.raw_to_json(self.get_response(rack_img, rack_analyzer_prompt))

        return response

    def classify_object(self, object_img):
        """
        Takes the current pic of the object of interest along with the current data of the rack and determines where
        to place the object

        :return:
        """
        prompt = self.semantic_classifier_prompt.replace(
    "{{RACK_AUDIT_JSON_FROM_PASS_1}}", json.dumps(self.data, indent=2))

        response = self.raw_to_json(self.get_response(object_img, prompt))
        print("Gemini Response:", response.text)

        return response

    def run(self):
        """
        Get images of rack and object and return target location for the object
        :return:
        """
        rack_img = self.get_image(self.get_realsense_frame())
        time.sleep(5)
        object_img = self.get_image(self.get_realsense_frame())

        if rack_img is not None and object_img is not None:
            self.data = self.analyze_rack(rack_img)
            obj_info = self.classify_object(object_img)

            return obj_info["chosen_location"]

        else:
            return None


    def raw_to_json(self, raw):
        raw = raw.text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        return parsed

    def get_realsense_frame(self):
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

    def get_image(self, raw_image):
        if raw_image is not None:
            # Convert numpy array to bytes for Gemini
            # Gemini expects RGB, but OpenCV/RealSense usually uses BGR
            import cv2
            from PIL import Image

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

        else:
            return None

        return pil_image


if __name__ == "__main__":
    pass
