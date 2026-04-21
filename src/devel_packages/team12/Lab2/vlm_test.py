import os
import time
import json
import re
import base64
import numpy as np
import cv2
import requests
import pyrealsense2 as rs

# --- Prompts ---

RACK_ANALYZER_PROMPT = """
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

SEMANTIC_CLASSIFIER_PROMPT = """
You are a robotic librarian. Below is the current semantic state of the rack and an image of a new object that needs to be placed.

Current Rack Audit:
{{RACK_AUDIT_JSON}}

Task:
1. Analyze the semantic relationship between the new object and the existing categories/themes in the rack.
2. Select the optimal location for the new object to maintain logical organization. 
3. If the rack is empty, pick a logical starting position (e.g., 'top-left').
4. If the rack is partially full, prioritize empty slots that align with the new object's semantic theme.

Output ONLY valid JSON:
{
  "chosen_location": "...",
  "reasoning": "Explain why this location is semantically aligned with the new object's category and theme."
}
"""

# --- Main Class ---

class vlm_autonomy:
    def __init__(self):
        # API Configuration
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY environment variable is empty.")
            
        # Using Gemini 1.5 Flash via REST API for Python 3.8 compatibility
        self.model_id = "gemini-2.5-flash"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        
        self.data = {}

    def get_response_from_api(self, b64_image, prompt):
        """
        Sends the prompt and image to Gemini via standard REST API.
        """
        if not b64_image:
            print("Error: No image provided to get_response_from_api")
            return None

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"HTTP Request Error: {e}")
            return None

    def analyze_rack(self, b64_img):
        print("Auditing rack contents...")
        raw_response = self.get_response_from_api(b64_img, RACK_ANALYZER_PROMPT)
        return self.parse_json_from_response(raw_response)

    def classify_object(self, b64_img):
        print("Determining best placement for object...")
        # Inject the current rack state into the prompt
        formatted_prompt = SEMANTIC_CLASSIFIER_PROMPT.replace(
            "{{RACK_AUDIT_JSON}}", 
            json.dumps(self.data, indent=2)
        )
        raw_response = self.get_response_from_api(b64_img, formatted_prompt)
        return self.parse_json_from_response(raw_response)

    def run(self, rack_frame, obj_frame):
        """
        Main execution loop for the VLM autonomy task.
        """
        # 1. Capture current Rack state
        rack_b64 = self.encode_image_to_base64(rack_frame)
        
        if not rack_b64:
            print("Failed to capture or encode rack image.")
            return None

        self.data = self.analyze_rack(rack_b64)
        if not self.data:
            print("Failed to parse rack audit data.")
            return None
        
        # 2. Capture the new object to be placed
        obj_b64 = self.encode_image_to_base64(obj_frame)
        
        if not obj_b64:
            print("Failed to capture or encode object image.")
            return None

        # 3. Get placement decision
        decision = self.classify_object(obj_b64)

        if decision:
            print("\n" + "="*30)
            print(f"DECISION: {decision.get('chosen_location')}")
            print(f"REASON: {decision.get('reasoning')}")
            print("="*30 + "\n")
            return decision.get("chosen_location")
        
        return None

    def parse_json_from_response(self, response_json):
        """
        Extracts and cleans JSON from the Gemini REST response.
        Handles cases where the model includes conversational text around the JSON.
        """
        if not response_json:
            return None
            
        try:
            # Navigate the REST API response structure
            text_content = response_json['candidates'][0]['content']['parts'][0]['text']
            
            # Find the first '{' and the last '}' to isolate the JSON object
            start_index = text_content.find('{')
            end_index = text_content.rfind('}')
            
            if start_index == -1 or end_index == -1:
                print("Error: No JSON object found in the response text.")
                print(f"Raw Text: {text_content}")
                return None
                
            # Extract only the substring containing the JSON
            json_str = text_content[start_index:end_index + 1]
            
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON Extraction Error: {e}")
            if 'text_content' in locals():
                print(f"Raw Text received: {text_content}")
            return None

    def encode_image_to_base64(self, numpy_image):
        """
        Converts a BGR numpy array to a Base64-encoded JPEG string.
        """
        if numpy_image is None:
            return None
        
        success, buffer = cv2.imencode('.jpg', numpy_image)
        if not success:
            return None
            
        return base64.b64encode(buffer).decode('utf-8')

if __name__ == "__main__":
    # Example local test
    # vlm = vlm_autonomy()
    # vlm.run()
    pass