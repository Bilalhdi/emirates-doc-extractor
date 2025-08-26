#importing the required libraries
import cv2
import easyocr
import numpy as np
import base64
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from pydantic import BaseModel
from typing import Optional
import time
import tempfile
from datetime import datetime
import json
import os
from PIL import Image
import uvicorn
import fitz  # PyMuPDF
import re
import together
from dotenv import load_dotenv
import uuid
import io
import re
import logging
import warnings
import google.generativeai as genai
import pillow_heif

from dotenv import load_dotenv
load_dotenv()

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n-emirates.pt")

pillow_heif.register_heif_opener()

warnings.filterwarnings("ignore")

#ACTIVATE USING conda activate docreader
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the model for incoming JSON data
class FileData(BaseModel):
    data: str
    ext: str

# Pydantic model for the request body
class Base64Request(BaseModel):
    data: str
    extension: str

# FastAPI instance
app = FastAPI(
    title="Emirates ID Reader",
    description="This API can extract data from Emirates ID, Driving Licence, Vehicle Licence, Trade, and Pass documents."
)

#function to conver .heic to .png
def heic_to_png(image_path):
    """
    Converts .heic to .png, returns new path if conversion done,
    else returns the original path unchanged.
    """
    if image_path.lower().endswith('.heic'):
        img = Image.open(image_path)
        png_path = os.path.splitext(image_path)[0] + ".png"
        img.save(png_path, format="PNG")
        return png_path
    return image_path

# Function to convert .jfif images to .jpg
def jfif_to_jpg(image_path: str) -> str:
    """
    Converts .jfif to .jpg, returns new path if conversion done,
    else returns the original path unchanged.
    """
    if image_path.lower().endswith('.jfif'):
        try:
            img = Image.open(image_path)
            # Ensure it's RGB (drop alpha if present)
            if img.mode == "RGBA":
                img = img.convert("RGB")

            jpg_path = os.path.splitext(image_path)[0] + ".jpg"
            img.save(jpg_path, format="JPEG")
            return jpg_path
        except Exception as e:
            print(f"Error converting {image_path} to JPG: {e}")
            return image_path
    return image_path


# Function to classify the document, than cropped and rotated the document and send it to it's respective model
def process_file(file_path: str, model_path: str = YOLO_WEIGHTS_PATH, cropped_dir: str = 'cropped_images', oriented_dir: str = 'oriented_images'):
    """
    Processes the given file (PDF or image), detects objects using the YOLO model,
    crops the detected regions, and corrects the orientation of the cropped images.

    Args:
        file_path (str): The path to the PDF or image file.
        model_path (str): The path to the YOLO model (default: r"YOLO_WEIGHTS_PATH").
        cropped_dir (str): The directory to save cropped images (default: 'cropped_images').
        oriented_dir (str): The directory to save oriented images (default: 'oriented_images').

    Returns:
        List of paths to the processed images.
    """

    # Load the trained YOLO model
    model_classify = YOLO(model_path)

    # Create directories for saving cropped and oriented images
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(oriented_dir, exist_ok=True)

    # Rotation map to correct orientations
    rotation_map = {
        '0': 0,
        '90': 270,  # Rotating 270 degrees is equivalent to rotating -90 degrees
        '180': 180,
        '270': 90,  # Rotating 90 degrees is equivalent to rotating -270 degrees
    }

    def process_pdf(pdf_path, dpi=300):
        """
        Convert each page of the PDF into a high-quality image using PyMuPDF.
        Ensures unique identification of each page.
        
        Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Dots per inch for image resolution (default: 300)
        
        Returns:
        list: Paths to the extracted images with page numbers
        """
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Create a timestamp for this batch processing to ensure uniqueness
        batch_id = int(time.time())
        
        for i in range(len(doc)):
            page = doc[i]
            
            # Set the matrix for higher resolution
            zoom = dpi / 72  # 72 is the default PDF resolution
            mat = fitz.Matrix(zoom, zoom)
            
            # Get the pixmap using the matrix for higher resolution
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save the image with high quality - include batch_id and page number in filename
            img_path = f"{os.path.splitext(pdf_path)[0]}_batch_{batch_id}_page_{i + 1}.png"
            img.save(img_path, format="PNG", dpi=(dpi, dpi), quality=95)
            
            image_paths.append(img_path)
        
        doc.close()
        return image_paths


    def process_image(image_path):
        """
        Process a single image for detection, cropping, and orientation correction.
        Uses unique identifiers to prevent filename conflicts.
        """
        # Extract page information from filename if it exists
        page_info = ""
        if "_page_" in image_path:
            # Extract page number from the filename
            match = re.search(r'_batch_(\d+)_page_(\d+)', image_path)
            if match:
                batch_id, page_num = match.groups()
                page_info = f"_batch{batch_id}_p{page_num}"
        
        # Generate a unique document ID for this processing session
        doc_uuid = uuid.uuid4().hex[:8]
        
        results = model_classify(source=image_path, save=True, conf=0.55)
        processed_images = []
        
        for i, result in enumerate(results):
            img = Image.open(result.path)
            # Convert the image to RGB mode if it has an alpha channel
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            for j, box in enumerate(result.boxes.xyxy):
                class_idx = int(result.boxes.cls[j].item())
                class_name = result.names[class_idx]
                confidence = result.boxes.conf[j].item()
                confidence = round(confidence, 2)

                # Extract document type, side, and orientation from the class name
                parts = class_name.split('_')
                
                if len(parts) == 3:
                    doc_type, side, orient = parts
                    # Save cropped and oriented images with proper naming including UUID and page info
                    xmin, ymin, xmax, ymax = map(int, box)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    
                    # Include UUID and page info in the filename to ensure uniqueness
                    cropped_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_cropped.jpg'
                    cropped_img_path = os.path.join(cropped_dir, cropped_img_name)
                    cropped_img.save(cropped_img_path)
                    processed_images.append(cropped_img_path)

                    if orient in rotation_map:
                        rotation_angle = rotation_map[orient]
                        if rotation_angle != 0:
                            cropped_img = cropped_img.rotate(rotation_angle, expand=True)

                    oriented_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_oriented.jpg'
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    cropped_img.save(oriented_img_path)
                    processed_images.append(oriented_img_path)

                else:
                    doc_type, orient = parts[0], parts[1]
                    side = 'front'  # No side information for certificates

                    # Include UUID and page info in the filename to ensure uniqueness
                    non_cropped_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_non_cropped.jpg'
                    non_cropped_img_path = os.path.join(cropped_dir, non_cropped_img_name)
                    img.save(non_cropped_img_path)
                    processed_images.append(non_cropped_img_path)
                    
                    # Save the image as it is in oriented_dir (no rotation)
                    oriented_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_oriented.jpg'
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    img.save(oriented_img_path)
                    processed_images.append(oriented_img_path)

        return processed_images
    processed_files = []
    if file_path.endswith('.pdf'):
        image_paths = process_pdf(file_path)
        for img_path in image_paths:
            processed_files.extend(process_image(img_path))
    else:
        processed_files.extend(process_image(file_path))

    return processed_files




class DocumentOCRProcessor:
    def __init__(self, api_key, model="gemini-2.5-flash-lite"):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.classifier = YOLO(YOLO_WEIGHTS_PATH)  # <-- Initialize YOLO for fallback
        print(f"DocumentOCRProcessor initialized with model: {self.model.model_name}")

        
    def _process_image_with_prompt(self, image_path, prompt, max_tokens=2048):
        """Helper to send image + prompt to Gemini and get text response"""
        try:
            logger.info(f"Processing image for Gemini: {image_path}")
            img = Image.open(image_path)
            generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=0.0)
            response = self.model.generate_content([prompt, img], generation_config=generation_config, stream=False)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return ""



    def encode_image(self, image_path):
        """
        Encode image to base64
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')



    def _extract_json_from_response(self, response: str):
        """
        Extracts JSON from LLM response safely.
        - Removes ```json fences
        - Handles stray commas and whitespace
        - Falls back to key-value parsing if JSON fails
        """
        import re

        # Log raw response for debugging
        logger.info(f"Raw VLM response: {response}")

        if not response or not response.strip():
            logger.warning("Empty response from LLM")
            return {}

        # 1️⃣ Remove markdown fences (```json ... ```)
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # 2️⃣ Attempt JSON parse
        try:
            parsed_json = json.loads(cleaned)
            if isinstance(parsed_json, dict):
                logger.info(f"Successfully parsed JSON: {parsed_json}")
                return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}, attempting fallback parsing...")

        # 3️⃣ Fallback: extract "key": "value" pairs manually
        constructed_data = {}
        for line in cleaned.splitlines():
            if ':' not in line:
                continue
            key_val = line.split(':', 1)
            if len(key_val) == 2:
                key = key_val[0].strip().strip('"').strip("'").strip(',')
                val = key_val[1].strip().strip('"').strip("'").strip(',')
                if key:
                    constructed_data[key] = val if val.lower() != 'null' else None

        if constructed_data:
            logger.info(f"Fallback parsing yielded: {constructed_data}")
            return constructed_data

        logger.warning("All parsing attempts failed, returning empty dict")
        return {}
    def detect_emiratesid_front(self, image_path):
        """Determine if Emirates ID FRONT is new or old format."""
        prompt = (
            "SYSTEM: Determine if this is an Emirates ID FRONT (new) or FRONT (old).\n"
            "Rules:\n"
            "- NEW FRONT shows: Name, Emirates ID, Nationality, Date of Birth, Expiry Date (often Issuing Date too) on the FRONT.\n"
            "- OLD FRONT shows: Name, Emirates ID, Nationality only (DOB & Expiry are NOT on the front).\n"
            "- If Date of Birth, Expiry date PARSED OR FOUND, it is the new format. if these words are NOT PARSED, label as old format\n"

            "Return ONLY 'new' or 'old'."
        )
        gemini_response = self._process_image_with_prompt(image_path, prompt, max_tokens=10)
        decision = (gemini_response or "").strip().lower()
        logger.info(f"Gemini (front) decision: {decision}")

        if "new" in decision:
            return "new"
        elif "old" in decision:
            return "old"

        # Fallback: YOLO class name (expects labels to encode new/old if available)
        try:
            results = self.classifier(image_path)
            for r in results:
                if hasattr(r, "names") and hasattr(r, "boxes") and len(r.boxes.cls) > 0:
                    class_id = int(r.boxes.cls[0].item())
                    label = r.names[class_id].lower()
                    # examples that would work: "emiratesid_front_new", "emiratesid_front_old"
                    if "front" in label:
                        if "new" in label:
                            return "new"
                        if "old" in label:
                            return "old"
        except Exception as e:
            logger.warning(f"YOLO fallback (front) failed: {e}")

        return "unknown"


    def process_emiratesid_front_old(self, image_path):
        """
        Parse OLD FRONT:
        - EID Customer Name
        - Emirates ID
        - EID Nationality
        """
        prompt = """
    SYSTEM: You are a strict JSON extractor. From an OLD-FORMAT Emirates ID FRONT image,
    return ONLY this JSON object (no prose, no markdown):

    {
    "Type": "Emirates ID Frontside",
    "EID Customer Name": <full English name or transliteration>,
    "Emirates ID": <id with dashes if present>,
    "EID Nationality": <nationality in English>
    }

    Rules:
    - Use double quotes around keys and string values.
    - If any value is missing or unreadable, use null.
    - Output ONLY a valid JSON object.
    """
        try:
            base64_img = self.encode_image(image_path)
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.0
            )
            response = self.model.generate_content([prompt, img], generation_config=generation_config, stream=False)
            raw_text = response.text
            json_data = self._extract_json_from_response(raw_text)
            return json_data
        except Exception as e:
            logger.error(f"Error processing Emirates ID front (old): {e}")
            return {}


    def process_emiratesid_front_new(self, image_path):
        """
        Parse NEW FRONT:
        - EID Customer Name
        - Emirates ID
        - EID Nationality
        - EID Date of Birth
        - EID Expiry date
        """
        prompt = """
    SYSTEM: You are a strict JSON extractor. From a NEW-FORMAT Emirates ID FRONT image,
    return ONLY this JSON object (no prose, no markdown):

    {
    "Type": "Emirates ID Frontside",
    "EID Customer Name": <full English name or transliteration>,
    "Emirates ID": <id with dashes if present>,
    "EID Nationality": <nationality in English>,
    "EID Date of Birth": <dd/mm/yyyy>,
    "EID Expiry date": <dd/mm/yyyy>
    }

    Rules:
    - Normalize dates to dd/mm/yyyy when possible.
    - Use double quotes around keys and string values.
    - If any value is missing or unreadable, use null.
    - Output ONLY a valid JSON object.
    """
        try:
            base64_img = self.encode_image(image_path)
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content([prompt, img], generation_config=generation_config, stream=False)
            raw_text = response.text
            json_data = self._extract_json_from_response(raw_text)
            return json_data
        except Exception as e:
            logger.error(f"Error processing Emirates ID front (new): {e}")
            return {}


    def detect_emiratesid_back(self, image_path):
        """Determine if Emirates ID BACK is new or old format."""
        prompt = (
            "SYSTEM: Determine if this is an Emirates ID BACK (new) or BACK (old).\n"
            "Rules:\n"
            "- NEW BACK shows Occupation, Employer, Issuing Place (bilingual).\n"
            "- OLD BACK typically shows MRZ chevrons '<<<' and usually lacks explicit Occupation/Employer lines.\n"
            "Return ONLY 'new' or 'old'."
            "If you see, parse the words Occupation, Employer, Issuing Place, then it is new format. If you see the words Date of Birth, Expiry date, then it is old format. Do the labelling for old and new accordingly.\n"

        )
        gemini_response = self._process_image_with_prompt(image_path, prompt, max_tokens=10)
        decision = (gemini_response or "").strip().lower()
        logger.info(f"Gemini (back) decision: {decision}")

        if "new" in decision:
            return "new"
        elif "old" in decision:
            return "old"

        # Fallback: YOLO class name (expects labels to encode new/old if available)
        try:
            results = self.classifier(image_path)
            for r in results:
                if hasattr(r, "names") and hasattr(r, "boxes") and len(r.boxes.cls) > 0:
                    class_id = int(r.boxes.cls[0].item())
                    label = r.names[class_id].lower()
                    # examples that would work: "emiratesid_back_new", "emiratesid_back_old"
                    if "back" in label:
                        if "new" in label:
                            return "new"
                        if "old" in label:
                            return "old"
        except Exception as e:
            logger.warning(f"YOLO fallback (back) failed: {e}")

        return "unknown"


    def process_emiratesid_back_old(self, image_path):
        """
        Parse OLD BACK:
        - EID Date of Birth
        - EID Expiry date
        """
        prompt = """
    SYSTEM: You are a strict JSON extractor. From an OLD-FORMAT Emirates ID BACK image,
    return ONLY this JSON object (no prose, no markdown):

    {
    "Type": "Emirates ID Backside",
    "EID Date of Birth": <dd/mm/yyyy>,
    "EID Expiry date": <dd/mm/yyyy>
    }

    Rules:
    - Normalize dates to dd/mm/yyyy when possible.
    - Use double quotes around keys and string values.
    - If any value is missing or unreadable, use null.
    - Output ONLY a valid JSON object.
    """
        try:
            base64_img = self.encode_image(image_path)
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=600,
                temperature=0.0
            )
            response = self.model.generate_content([prompt, img], generation_config=generation_config, stream=False)
            raw_text = response.text
            json_data = self._extract_json_from_response(raw_text)
            return json_data
        except Exception as e:
            logger.error(f"Error processing Emirates ID back (old): {e}")
            return {}


    def process_emiratesid_back_new(self, image_path):
        """
        Parse NEW BACK:
        - EID Occupation
        - EID Issuing Place
        - EID Employer
        """
        prompt = """
    SYSTEM: You are a strict JSON extractor. From a NEW-FORMAT Emirates ID BACK image,
    return ONLY this JSON object (no prose, no markdown):

    {
    "Type": "Emirates ID Backside",
    "EID Occupation": <occupation in English; translate Arabic>,
    "EID Issuing Place": <UAE city; translate Arabic>,
    "EID Employer": <employer name; translate if clearly Arabic>
    }

    Rules:
    - Use double quotes around keys and string values.
    - If any value is missing or unreadable, use null.
    - Output ONLY a valid JSON object.
    """
        try:
            base64_img = self.encode_image(image_path)
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.0
            )
            response = self.model.generate_content([prompt, img], generation_config=generation_config, stream=False)
            raw_text = response.text
            json_data = self._extract_json_from_response(raw_text)
            return json_data
        except Exception as e:
            logger.error(f"Error processing Emirates ID back (new): {e}")
            return {}


    # def process_emirates_id_front(self, image_path):
    #     """
    #     Process Emirates ID image using Gemini LLM

    #     Args:
    #         image_path (str): Path to the Emirates ID image

    #     Returns:
    #         dict: Extracted Emirates ID information
    #     """
    #     prompt = """
    #     SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    #     TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    #     - Name (should be named EID Customer Name in json output)
    #     - ID Number (should be named Emirates ID in json output)
    #     - Date of Birth (should be named EID Date of Birth in json output)
    #     - Expiry Date (should be named EID Expiry date in json output)
    #     - Nationality (should be named EID Naitionality in json output)
    #     - EID Customer Name (this is the same as Name, but should be in the key "EID Customer Name")

    #     FORMAT REQUIREMENTS:
    #     1. Output MUST be a single JSON object starting with "{"
    #     2. Use double quotes for ALL keys and string values
    #     3. Use null for missing or unreadable values
    #     4. Do NOT include any text outside the JSON object
    #     5. Do NOT use markdown or "json" markers
    #     6. Type of doc (should be Emirates ID FrontSide always)

    #     EXAMPLE CORRECT OUTPUT:
    #     {
    #     "Type": "Emirates ID FrontSide",  
    #     "EID Customer Name": "John Doe", 
    #     "Emirates ID": "1234567890123456",
    #     "EID Date of Birth": "01/01/1980",
    #     "EID Expiry date": "01/01/2030",
    #     "EID Nationality": "USA"
    #     }

    #     WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    #     """

    #     try:
    #         # Encode to base64 for consistency
    #         base64_img = self.encode_image(image_path)

    #         # Convert base64 to PIL Image for Gemini
    #         from io import BytesIO
    #         img_bytes = base64.b64decode(base64_img)
    #         img = Image.open(BytesIO(img_bytes))

    #         # Gemini call
    #         generation_config = genai.types.GenerationConfig(
    #             max_output_tokens=1000,
    #             temperature=0.0
    #         )
    #         response = self.model.generate_content(
    #             [prompt, img],
    #             generation_config=generation_config,
    #             stream=False
    #         )

    #         raw_text = response.text
    #         print(f"Model Response:\n{raw_text}")

    #         # Extract JSON
    #         json_response = self._extract_json_from_response(raw_text)
    #         print(f"Extracted JSON:\n{json_response}")
    #         return json_response

    #     except Exception as e:
    #         print(f"Error processing Emirates ID: {e}")
    #         return {}


    # def process_emirates_id_back(self, image_path):
    #     """
    #     Process Emirates ID (back side) using Gemini LLM

    #     Args:
    #         image_path (str): Path to the Emirates ID back image

    #     Returns:
    #         dict: Extracted Emirates ID back information
    #     """
    #     prompt = """
    #     SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, 
    #     NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    #     TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    #     - Emirates ID (Card Number: 9 digits at top left, choose ONLY the first 9 digits if longer)
    #     - Occupation (should be named EID Occupation in json output)
    #     - Issuing Place (should be named EID Issuing Place in json output)
    #     - Employer

    #     FORMAT REQUIREMENTS:
    #     1. Output MUST be a single JSON object starting with '{' and ending with '}'
    #     2. Use double quotes for ALL keys and string values
    #     3. Use null for missing or unreadable values
    #     4. Do NOT include any text outside the JSON object
    #     5. Do NOT use markdown or "json" markers
    #     6. Type of doc (should be Emirates ID Back Side always)

    #     EXAMPLE CORRECT OUTPUT:
    #     {
    #     "Type": "Emirates ID Back Side",  
    #     "Emirates ID": "123456789",
    #     "EID Occupation": "Engineer",
    #     "EID Issuing Place": "Dubai",
    #     "EID Employer": "Safe Water Technology L.L.C"
    #     }

    #     WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    #     """

    #     try:
    #         # Encode the image to base64 for consistency
    #         base64_img = self.encode_image(image_path)

    #         # Convert base64 to PIL Image for Gemini
    #         from io import BytesIO
    #         img_bytes = base64.b64decode(base64_img)
    #         img = Image.open(BytesIO(img_bytes))

    #         # Call Gemini
    #         generation_config = genai.types.GenerationConfig(
    #             max_output_tokens=1000,
    #             temperature=0.0
    #         )
    #         response = self.model.generate_content(
    #             [prompt, img],
    #             generation_config=generation_config,
    #             stream=False
    #         )

    #         raw_text = response.text
    #         print(f"Model Response:\n{raw_text}")

    #         # Extract JSON from the model response
    #         json_response = self._extract_json_from_response(raw_text)
    #         print(f"Extracted JSON:\n{json_response}")

    #         return json_response

        except Exception as e:
            print(f"Error processing Emirates ID back: {e}")
            return {}
    def detect_vehicle_license_type(self, image_path):
        """Determine if license is electronic or card"""
        prompt = (
            "SYSTEM: Determine if the image is an electronic UAE vehicle license or a card.\n"
            "Rules: Electronic usually has QR codes or black/white layout. Card has eagle/bird logo, color, no QR.\n"
            "Return ONLY 'electronic' or 'card'."
        )
        gemini_response = self._process_image_with_prompt(image_path, prompt, max_tokens=10)
        decision = gemini_response.strip().lower()
        logger.info(f"Gemini decision: {decision}")

        if "electronic" in decision:
            return "electronic"
        elif "card" in decision:
            return "card"

        # fallback: YOLO classification
        results = self.classifier(image_path)
        for r in results:
            if hasattr(r, "names") and hasattr(r, "boxes") and len(r.boxes.cls) > 0:
                class_id = int(r.boxes.cls[0].item())
                label = r.names[class_id].lower()
                if "electronic" in label:
                    return "electronic"
                elif "card" in label or "vehicle" in label:
                    return "card"
        return "unknown"    
    def process_electronic_vehicle_license(self, image_path):
        prompt = """
SYSTEM: You are a JSON extraction AI. Extract the following from an electronic UAE vehicle license image and return ONLY a JSON object:
- Traffic plate no.
- Place of Issue (sometimes will be in arabic, translate to english and add it to the json) (place of issue will always be a city in UAE)
- Plate class
- M Traffic Id T.C No
- M Customer Name   (sometimes might be company name, read it and add it to the json and sometimes might be in arabic, translate to english and add it to the json)
- Company 
- M Ins Expiry
- Nationality
- Policy No.
- M Issue Date (registration date)
- M Expiry Date
- Insurance Type
- Mortgaged by
- Insurance Company
- No. of passengers
- Year model
- Origin
- Vehicle color
- Vehicle class
- Motor Make Desc/Veh type (This will be the vehicle type's first word, brand of car)
- Variant Desc Model (This will be vehicle type's second word, model of car)
- Empty weight
- Gross vehicle weight
- Engine No.
- M chasis No
- Type of doc (Should be Mulkiya Frontside Electronic always)
Return only a valid JSON object, no extra text or formatting, use null for missing/unreadable values. At top add "Type": "Mulkiya Frontside Electronic", followed by the rest of the json outputs.
"""
        try:
                # Keep your old base64 logic
            base64_img = self.encode_image(image_path)

            # Gemini expects image input as PIL Image, so decode base64 back to bytes
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Gemini call
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            json_data = self._extract_json_from_response(raw_text)
            return raw_text, json_data

        except Exception as e:
            print(f"Error processing electronic vehicle license: {e}")
            return "", {}
    

    def process_vehicle_licence_front(self, image_path):
        """
        Process Vehicle Licence front image using Gemini LLM
        
        Args:
            image_path (str): Path to the Vehicle Licence image
        
        Returns:
            dict: Extracted Vehicle Licence information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - Type of doc (should be Vehicle licence front)
    - M Traffic Id T.C No
    - M Customer Name (could be company name, translate Arabic to English if needed)
    - M Nationality
    - M Expiry Date
    - M Issue Date
    - M Ins Expiry
    - Insurance Company (if arabic write it as arabic text, dont translate)
    - Place of Issue (translate Arabic to English, always a UAE city)

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with "{" and ending with "}"
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown
    6. Do NOT include "json" or any markers
    7. Type of doc (should be Mulkiya Frontside always)
    8. Read Arabic headings/values. If a heading like "شركة التأمين" or "مؤمنة لدى" or similar appears, capture the value as Insurance Company

    EXAMPLE CORRECT OUTPUT:
    {
    "Type": "Mulkiya Frontside",
    "M Traffic Id T.C No": "12345678",
    "M Customer Name": "John Doe",
    "M Nationality": "USA",
    "M Expiry Date": "01/01/2030",
    "M Ins Expiry": "01/01/2025",
    "M Issue Date": "01/01/2020",
    "Place of Issue": "Dubai"
    "Insurance Company" (شركة التأمين) (مؤمنة لدى): "شركة دبي للتأمين (ش.م.ك)" (if arabic write it as arabic text)
    }

    WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Gemini expects PIL Image, so decode base64 into bytes and open
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Call Gemini
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")
            
            # Extract JSON
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")
            return json_response

        except Exception as e:
            print(f"Error processing Vehicle Licence: {e}")
            return {}


    def process_vehicle_licence_back(self, image_path):
        """
        Process Vehicle Licence back image using Gemini LLM
        
        Args:
            image_path (str): Path to the Vehicle Licence image
        
        Returns:
            dict: Extracted Vehicle Licence information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - Motor Make Desc/Veh type (This will be the vehicle type's first word, brand of car)
    - Variant Desc Model (This will be vehicle type's second word, model of car)
    - M Origin
    - M Veh. Type
    - M Engine No (Note: It will be written as NIL most of the time. It will not contain any unit like k.g or cm)
    - M chasis No

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with "{" and ending with "}"
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown
    6. Do NOT include "json" or any markers
    7. Type of doc (should be Mulkiya Backside always)
    8. In case Variant Desc Model not available, make it NIL

    EXAMPLE CORRECT OUTPUT:
    {
    "Type": "Mulkiya Backside
",  
    "Motor Make Desc/Veh type": "Toyota",
    "Variant Desc Model": "Corolla",
    "M Origin": "Japan",
    "M Veh. Type": "Sedan",
    "M Engine No": "NIL",
    "M chasis No": "ABC12345"
    }

    WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Gemini expects PIL Image, so decode base64 into bytes and open
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Call Gemini
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")

            # Extract JSON
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")
            return json_response

        except Exception as e:
            print(f"Error processing Vehicle Licence: {e}")
            return {}

    
    
    
    def process_driving_licence_front(self, image_path):
        """
        Process Driving Licence front image using Gemini LLM
        
        Args:
            image_path (str): Path to the Driving Licence image
        
        Returns:
            dict: Extracted Driving Licence information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Extract these fields from a UAE driving licence (front). 
    Your output MUST be a valid JSON object with NO markdown, NO code blocks, NO explanatory text, 
    and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - DL Customer Name (sometimes organization/company)
    - DL Date Of Birth
    - DL Expiry Date
    - DL Issue Date
    - License No
    - DL Nationality
    - Place of Issue (always a city in UAE)

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with '{' and ending with '}'
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown or JSON markers
    6. Type of doc (should be Driving License Frontside always)

    EXAMPLE:
    {
    "Type": "Driving License Frontside",
    "DL Customer Name": "John Doe",
    "DL Date Of Birth": "01/01/1980",
    "DL Expiry Date": "01/01/2030",
    "DL Issue Date": "01/01/2020",
    "License No": "ABC12345",
    "DL Nationality": "USA",
    "Place of Issue": "Dubai"
    }
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Convert base64 to PIL Image for Gemini
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Gemini call
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")

            # Extract JSON
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")

            return json_response

        except Exception as e:
            print(f"Error processing Driving Licence: {e}")
            return {}

        



    def process_driving_licence_back(self, image_path):
        """
        Process Driving Licence back image using Gemini LLM
        
        Args:
            image_path (str): Path to the Driving Licence image
        
        Returns:
            dict: Extracted Driving Licence information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, 
    NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - DL Traffic Id T.C No (Traffic Id T.C No is a 9 digit number)

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with '{' and ending with '}'
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown or JSON markers
    6. Type of doc (should be Driving License Back Side always)

    EXAMPLE:
    {
    "Type": "Driving License Back Side",
    "DL Traffic Id T.C No": "567890123"
    }

    WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Convert base64 to PIL Image for Gemini
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Gemini call
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")

            # Extract JSON
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")

            return json_response

        except Exception as e:
            print(f"Error processing Driving Licence: {e}")
            return {}



    def process_pass_certificate(self, image_path):
        """
        Process Pass Certificate image using Gemini LLM
        
        Args:
            image_path (str): Path to the Pass Certificate image
        
        Returns:
            dict: Extracted Pass Certificate information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, 
    NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - Inspection Date (should be named Passing Date in json output)
    - Pass/Not Pass result (should be named Result Test Certificate in json output)

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with '{' and ending with '}'
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown or JSON markers
    6. Type of doc (should be Pass Certificate always)
    7. If any form like No Claim Self Declaration Form or Local Purcharse Order or anything that is not a Pass Certificate, return null for Result Test Certificate

    EXAMPLE CORRECT OUTPUT:
    {
    "Type": "Test Certificate",   (SHOULD ALWAYS BE Test Certificate NO MATTER WHAT IS WRITTEN ON IMAGE)
    "Passing Date": "01/01/2022",
    "Result Test Certificate": "Pass" | "Fail" | null (anyone of these 3 found WRITTEN ON IMAGE)
    }

    WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Gemini expects PIL image input
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Gemini call
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")

            # Extract JSON from response
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")

            return json_response

        except Exception as e:
            print(f"Error processing Pass Certificate: {str(e)}")
            return {}


    def process_trade_certificate(self, image_path):
        """
        Process Trade Certificate image using Gemini LLM
        
        Args:
            image_path (str): Path to the Trade Certificate image
        
        Returns:
            dict: Extracted Trade Certificate information
        """
        prompt = """
    SYSTEM: You are a JSON extraction AI. Your output MUST be a valid JSON object with NO markdown, 
    NO code blocks, NO explanatory text, and NO formatting like **bold** or *italic*.

    TASK: Extract the following fields from the provided image and return ONLY a valid JSON object:
    - Company Name
    - Issue Date (should be named TL Issue Date in json output)
    - Expiry Date (should be named TL Expiry Date in json output)
    - Activity (should be named Trade Activity in json output)
    - TL Number (Trade License Trade number, will be 6 digits)
    

    FORMAT REQUIREMENTS:
    1. Output MUST be a single JSON object starting with '{' and ending with '}'
    2. Use double quotes for ALL keys and string values
    3. Use null for missing or unreadable values
    4. Do NOT include any text outside the JSON object
    5. Do NOT use markdown or JSON markers
    6. Type of doc (should be Trade License always, not economic license or anything else on the image, always Trade License)
    7. If any form like No Claim Self Declaration Form or Local Purcharse Order or anything that is not a Trade License, return null for TL Issue Date, TL Expiry Date, Trade Activity and TL Number

    EXAMPLE CORRECT OUTPUT:
    { 
    "Type": "Trade License", (SHOULD ALWAYS BE Trade License NO MATTER WHAT IS WRITTEN ON IMAGE)
    "Company Name": "ABC Trading",
    "TL Issue Date": "01/01/2022",
    "TL Expiry Date": "01/01/2023",
    "Trade Activity": "Retail"
    "TL Number": "123456789" (will be 6 digits)
    }

    WARNING: Any deviation from a valid JSON object will cause a system error. Output ONLY the JSON object.
    """

        try:
            # Keep old base64 logic
            base64_img = self.encode_image(image_path)

            # Convert base64 to PIL image for Gemini
            from io import BytesIO
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(BytesIO(img_bytes))

            # Gemini call
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=generation_config,
                stream=False
            )

            raw_text = response.text
            print(f"Model Response:\n{raw_text}")

            # Extract JSON
            json_response = self._extract_json_from_response(raw_text)
            print(f"Extracted JSON:\n{json_response}")

            return json_response

        except Exception as e:
            print(f"Error processing Trade Certificate: {str(e)}")
            return {}
ocr_processor = DocumentOCRProcessor(api_key=GEMINI_API_KEY)


def id_front(img_path):
    """Route Emirates ID FRONT to the correct parser (new vs old)."""
    fmt = ocr_processor.detect_emiratesid_front(img_path)  # 'new' | 'old' | 'unknown'
    logger.info(f"[ID FRONT] format detected: {fmt}")
    if fmt == "new":
        return ocr_processor.process_emiratesid_front_new(img_path)
    elif fmt == "old":
        return ocr_processor.process_emiratesid_front_old(img_path)
    else:
        # Fallback: try new, then old
        out = ocr_processor.process_emiratesid_front_new(img_path)
        return out or ocr_processor.process_emiratesid_front_old(img_path)

def id_back(img_path):
    """Route Emirates ID BACK to the correct parser (new vs old)."""
    fmt = ocr_processor.detect_emiratesid_back(img_path)  # 'new' | 'old' | 'unknown'
    logger.info(f"[ID BACK] format detected: {fmt}")
    if fmt == "new":
        return ocr_processor.process_emiratesid_back_new(img_path)
    elif fmt == "old":
        return ocr_processor.process_emiratesid_back_old(img_path)
    else:
        # Fallback: try new, then old
        out = ocr_processor.process_emiratesid_back_new(img_path)
        return out or ocr_processor.process_emiratesid_back_old(img_path)

def driving_front(img_path):
    return ocr_processor.process_driving_licence_front(img_path)

def driving_back(img_path):
    return ocr_processor.process_driving_licence_back(img_path)

def vehicle_front(img_path):
    return ocr_processor.process_vehicle_licence_front(img_path)
    
def vehicle_back(img_path):
    return ocr_processor.process_vehicle_licence_back(img_path)

def pass_certificate(img_path):
    return ocr_processor.process_pass_certificate(img_path)

def trade_certificate(img_path):
    return ocr_processor.process_trade_certificate(img_path)








@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF or image) and processes it to detect objects using the YOLO model,
    crops the detected regions, and corrects the orientation of the cropped images.
    Handles multiple documents of the same type across multiple pages.

    Args:
        file (UploadFile): The file to upload.

    Returns:
        JSONResponse: The response containing the extracted information from all detected documents.
    """
    start_time = time.time()
    
    try:
        # Save the uploaded file temporarily with a unique name
        timestamp = int(time.time())
        temp_file_path = f"temp_{timestamp}_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the file using the improved process_file method
        processed_files = process_file(temp_file_path)

        # Filter out only the oriented images
        oriented_files = [file for file in processed_files if 'oriented' in file]

        if not oriented_files:
            return JSONResponse(
                content={"error": "No valid documents detected in the file"},
                status_code=400
            )

        # Group oriented files by their unique identifiers
        # Instead of overwriting, we'll now process each detected document individually
        grouped_documents = []
        
        for oriented_file in oriented_files:
            file_name = os.path.basename(oriented_file)
            parts = file_name.split('_')
            
            # Extract document info from filename
            doc_type = parts[0]
            side = parts[1] if len(parts) >= 2 else 'front'
            
            # Extract unique document identifier (for internal use only)
            uuid_match = re.search(r'_([a-f0-9]{8})_\d+_\d+_', file_name)
            doc_id = uuid_match.group(1) if uuid_match else "unknown"
            
            # Extract confidence correctly from filename
            confidence_match = re.search(r'_(\d+\.\d+)_oriented\.jpg$', oriented_file)
            if confidence_match:
                confidence = confidence_match.group(1)
            else:
                confidence = "0.0"
                
            # Create document object
            document = {
                'file_path': oriented_file,
                'doc_type': doc_type,
                'side': side,
                'confidence': confidence,
                'doc_id': doc_id  # Keep for internal grouping but don't expose in response
            }
            
            grouped_documents.append(document)
            
        # Initialize results list
        image_results = []

        # Process each document individually
        for document in grouped_documents:
            oriented_file = document['file_path']
            doc_type = document['doc_type']
            side = document['side']
            confidence = document['confidence']
            print("Reading with OpenCV:", oriented_file)
            print("File exists?", os.path.exists(oriented_file))
            print("File size:", os.path.getsize(oriented_file) if os.path.exists(oriented_file) else "not found")

            # Read the image and calculate token usage
            img = cv2.imread(oriented_file)
            if img is None:
                print("OpenCV could not load the image!")
                continue
                
            img_np = np.array(img)
            
            image_height, image_width = img_np.shape[:2]
            tokens_used = (image_height * image_width) // 1000
            
            try:
                # Process based on document type
                if 'ID' in oriented_file:
                    if 'front' in oriented_file:
                        detected_info = id_front(oriented_file)
                    else:
                        detected_info = id_back(oriented_file)
                    
                elif 'Driving' in oriented_file:
                    if 'front' in oriented_file:
                        detected_info = driving_front(oriented_file)
                    else:
                        detected_info = driving_back(oriented_file)
                    
                elif 'vehicle' in oriented_file:
    # First detect license type
                    license_type = ocr_processor.detect_vehicle_license_type(oriented_file)
                    print(f"Detected vehicle license type: {license_type}")

                    if license_type == "electronic":
                        raw_text, json_data = ocr_processor.process_electronic_vehicle_license(oriented_file)
                        detected_info = json_data
                    else:
                        # fallback to physical card logic
                        if 'front' in oriented_file:
                            detected_info = vehicle_front(oriented_file)
                        else:
                            detected_info = vehicle_back(oriented_file)
                 
                elif 'pass' in oriented_file:
                    detected_info = pass_certificate(oriented_file)

                elif 'trade' in oriented_file:
                    detected_info = trade_certificate(oriented_file)
                else:
                    detected_info = {}
            except Exception as proc_error:
                detected_info = {"error": str(proc_error)}

            # Compile the result for the current image without document_id and page_info
            image_result = {
                "image_metadata": {
                    "Image_Path": oriented_file,
                    "Document_Type": doc_type,
                    "Confidence_score": confidence,  # Use correctly extracted confidence
                    "side": side,
                    "Tokens_Used": tokens_used
                },
                "detected_data": detected_info
            }

            # Append the result to the list of image results
            image_results.append(image_result)

        # Calculate overall processing time
        processing_time = time.time() - start_time

        # Compile the final response data
        # Extract ONLY the detected_data for each image (as a list)
        pure_json_results = [result['detected_data'] for result in image_results]
        return JSONResponse(content=pure_json_results)


    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


print("==> upload_base64 called")
@app.post("/upload/base64/")
@app.post("/upload/base64/")
async def upload_base64(request: Base64Request):
    import traceback
    from PIL import Image
    print("==> upload_base64 called")
    start_time = time.time()
    temp_file_path = None
    processed_files = []
    debug_file_path = None

    try:
        cleaned_data = request.data.strip()
        print("Received data length:", len(cleaned_data))

        # If it includes a data URI prefix, strip it
        if ',' in cleaned_data and ';base64,' in cleaned_data:
            cleaned_data = cleaned_data.split(',')[1]
        print("Cleaned base64 data, first 30 chars:", cleaned_data[:30])

        # Validate extension
        ext = request.extension.lower()
        print("Extension received:", ext)
        if ext not in ['pdf', 'png', 'jpg', 'jpeg', 'jfif', 'heic', 'heif']:
            print("Unsupported extension:", ext)
            return JSONResponse(
                content={"error": f"Unsupported file extension: {ext}"},
                status_code=400
            )

        # Decode the base64 data
        try:
            missing_padding = len(cleaned_data) % 4
            if missing_padding:
                cleaned_data += '=' * (4 - missing_padding)
            decoded_data = base64.b64decode(cleaned_data)
            print("Decoded base64, length:", len(decoded_data))
        except Exception as e:
            print("Base64 decode error!")
            print(traceback.format_exc())
            return JSONResponse(
                content={"error": f"Invalid base64 data: {str(e)}"},
                status_code=400
            )

        # Generate file paths with UUID to prevent collisions
        print("generating file paths")
        file_id = uuid.uuid4()
        temp_file_path = f"temp_base64_{file_id}.{ext}"
        debug_file_path = f"debug_base64_{file_id}.{ext}"

        # ---- Save the file to disk (no double saves) ----
        try:
            if ext in ['png', 'jpg', 'jpeg', 'jfif', 'heic', 'heif']:
                try:
                    img = Image.open(io.BytesIO(decoded_data))
                    # Normalize mode
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')

                    # Force .jfif to .jpg on disk (both temp & debug)
                    if ext in ['jpg', 'jpeg', 'jfif']:
                        temp_file_path  = temp_file_path.rsplit('.', 1)[0]  + '.jpg'
                        debug_file_path = debug_file_path.rsplit('.', 1)[0] + '.jpg'
                        save_format = "JPEG"
                    elif ext == 'png':
                        save_format = "PNG"
                    else:
                        # For heic/heif, save as PNG first; names keep .png here
                        temp_file_path  = temp_file_path.rsplit('.', 1)[0]  + '.png'
                        debug_file_path = debug_file_path.rsplit('.', 1)[0] + '.png'
                        save_format = "PNG"

                    img.save(temp_file_path,  format=save_format)
                    img.save(debug_file_path, format=save_format)
                    print("Image saved via PIL at", temp_file_path)
                except Exception as pil_exc:
                    print("PIL could not open or save, fallback to raw file:", pil_exc)
                    with open(temp_file_path, "wb") as buffer:
                        buffer.write(decoded_data)
                    with open(debug_file_path, "wb") as buffer:
                        buffer.write(decoded_data)
            else:
                # Non-image (e.g., PDF)
                with open(temp_file_path, "wb") as buffer:
                    buffer.write(decoded_data)
                with open(debug_file_path, "wb") as buffer:
                    buffer.write(decoded_data)
            print("Saved temp file at", temp_file_path, "size:", os.path.getsize(temp_file_path))
        except Exception as e:
            print("File write error!")
            print(traceback.format_exc())
            return JSONResponse(
                content={"error": f"Could not save uploaded file: {str(e)}"},
                status_code=500
            )

        # Normalize special cases
        temp_file_path = heic_to_png(temp_file_path)
        temp_file_path = jfif_to_jpg(temp_file_path)

        # Try opening the saved file with PIL to check for corruption
        if any(temp_file_path.lower().endswith(x) for x in ['.png', '.jpg', '.jpeg']):
            try:
                test_img = Image.open(temp_file_path)
                test_img.verify()
                print(f"{temp_file_path} is a valid image (PIL verified).")
            except Exception as verify_exc:
                print(f"PIL cannot verify {temp_file_path}: {verify_exc}")

        # YOLO + doc detection/cropping
        try:
            processed_files = process_file_with_checks(temp_file_path)
            print("Processed files:", processed_files)
        except Exception as e:
            print("Error in process_file_with_checks!")
            print(traceback.format_exc())
            return JSONResponse(
                content={"error": f"Error during YOLO/doc processing: {str(e)}"},
                status_code=500
            )

        oriented_files = [file for file in processed_files if 'oriented' in file]
        print("Oriented files:", oriented_files)

        if not oriented_files:
            print("No oriented files found!")
            return JSONResponse(
                content={"error": "No valid document detected in the image"},
                status_code=400
            )

        # Choose best per (doc_type, side)
        grouped_files = {}
        for oriented_file in oriented_files:
            file_name = os.path.basename(oriented_file)
            parts = file_name.split('_')
            doc_type = parts[0] if parts else ''
            side = parts[1] if len(parts) >= 2 else ('front' if 'front' in file_name.lower() else 'back')
            confidence_match = re.search(r'_(\d+\.\d+)_oriented\.(?:jpg|png)$', oriented_file, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            image_key = f"{doc_type}_{side}"
            if image_key not in grouped_files or confidence > grouped_files[image_key]['confidence']:
                grouped_files[image_key] = {
                    'file_path': oriented_file,
                    'confidence': confidence
                }

        image_results = []

        for image_key, image_data in grouped_files.items():
            oriented_file = image_data['file_path']
            print("Reading with OpenCV:", oriented_file)
            print("File exists?", os.path.exists(oriented_file))
            print("File size:", os.path.getsize(oriented_file) if os.path.exists(oriented_file) else "not found")

            try:
                img = cv2.imread(oriented_file)
                print("YOLO debug: cv2.imread() returned", type(img))
                if img is None:
                    print("Image failed to load in YOLO step for:", oriented_file)
                    continue
                img_np = np.array(img)
                image_height, image_width = img_np.shape[:2]
                tokens_used = (image_height * image_width) // 1000
            except Exception as e:
                print("cv2.imread or np.array error!")
                print(traceback.format_exc())
                continue

            # Parse doc_type/side again robustly
            file_name = os.path.basename(oriented_file)
            parts = file_name.split('_')
            doc_type = (parts[0] if parts else '').lower()
            side = (parts[1] if len(parts) >= 2 else ('front' if 'front' in file_name.lower() else 'back')).lower()
            confidence_match = re.search(r'_(\d+\.\d+)_oriented\.(?:jpg|png)$', oriented_file, re.IGNORECASE)
            confidence = confidence_match.group(1) if confidence_match else "0.0"

            # ---------- ROUTE TO THE RIGHT PARSER ----------
            try:
                if doc_type in ("id", "emiratesid", "eid", "emirates_id"):
                    if side == "front":
                        detected_info = id_front(oriented_file)
                    else:
                        detected_info = id_back(oriented_file)

                elif doc_type in ("driving", "drivinglicense", "drivinglicence", "dl"):
                    detected_info = driving_front(oriented_file) if side == "front" else driving_back(oriented_file)

                elif doc_type in ("vehicle", "mulkiya", "vehiclelicense", "vehiclelicence"):
                    # First detect license type
                    license_type = ocr_processor.detect_vehicle_license_type(oriented_file)
                    print(f"Detected vehicle license type: {license_type}")
                    if license_type == "electronic":
                        _, detected_info = ocr_processor.process_electronic_vehicle_license(oriented_file)
                    else:
                        detected_info = vehicle_front(oriented_file) if side == "front" else vehicle_back(oriented_file)

                elif doc_type in ("pass", "certificate", "test"):
                    detected_info = pass_certificate(oriented_file)

                elif doc_type in ("trade", "tradelicense", "tl"):
                    detected_info = trade_certificate(oriented_file)

                else:
                    detected_info = {}

            except Exception as proc_error:
                print("Error in document extraction!")
                print(traceback.format_exc())
                detected_info = {"error": str(proc_error)}
            # ---------- END ROUTING ----------

            image_result = {
                "image_metadata": {
                    "Image_Path": oriented_file,
                    "Document_Type": doc_type,
                    "Confidence_score": confidence,
                    "side": side,
                    "Tokens_Used": tokens_used if 'tokens_used' in locals() else None
                },
                "detected_data": detected_info
            }
            image_results.append(image_result)

        processing_time = time.time() - start_time

        response_data = {
            "overall_metadata": {
                "Total_PTime": f"{processing_time:.2f} seconds",
                "Total_Tokens_Used": sum([(res['image_metadata']['Tokens_Used'] or 0) for res in image_results]),
                "Images_Processed": len(image_results),
                "Timestamp": datetime.now().isoformat()
            },
            "images_results": image_results
        }

        print("Returning response")
        return JSONResponse(content=response_data)

    except Exception as e:
        print("Exception in main try block!")
        print(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    finally:
        print("Cleaning up temp files...")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if debug_file_path and os.path.exists(debug_file_path):
            os.remove(debug_file_path)
        try:
            for file_path in processed_files:
                if os.path.exists(file_path) and 'oriented' not in file_path:
                    os.remove(file_path)
        except Exception as clean_exc:
            print("Error during cleanup:", clean_exc)





# Add this new function to handle the file processing with additional checks
def process_file_with_checks(file_path: str, model_path: str = YOLO_WEIGHTS_PATH, cropped_dir: str = 'cropped_images', oriented_dir: str = 'oriented_images'):
    """Enhanced version of process_file that ensures proper coordinate handling and consistent confidence values"""

    # Load the trained YOLO model
    model_classify = YOLO(model_path)

    # Create directories for saving cropped and oriented images
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(oriented_dir, exist_ok=True)

    # Rotation map to correct orientations
    rotation_map = {
        '0': 0,
        '90': 270,
        '180': 180,
        '270': 90,
    }

    def process_pdf(pdf_path, dpi=300):
        """Process PDF with batch ID to ensure unique filenames"""
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Create a timestamp for this batch processing to ensure uniqueness
        batch_id = int(time.time())
        
        for i in range(len(doc)):
            page = doc[i]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_path = f"{os.path.splitext(pdf_path)[0]}_batch_{batch_id}_page_{i + 1}.png"
            img.save(img_path, format="PNG", dpi=(dpi, dpi), quality=95)
            image_paths.append(img_path)
        
        doc.close()
        return image_paths

    def process_image(image_path):
        """Enhanced process_image function with better coordinate handling and consistent confidence values"""
        # Extract batch and page info if present (for PDFs)
        page_info = ""
        if "_batch_" in image_path and "_page_" in image_path:
            match = re.search(r'_batch_(\d+)_page_(\d+)', image_path)
            if match:
                batch_id, page_num = match.groups()
                page_info = f"_batch{batch_id}_p{page_num}"
        
        # Generate a unique ID for this document processing
        doc_uuid = uuid.uuid4().hex[:8]
        
        # Load image with PIL first to get dimensions
        try:
            pil_img = Image.open(image_path)
            pil_img.close()
        except Exception:
            pass
        
        # Run YOLO detection
        results = model_classify(source=image_path, save=True, conf=0.55)
        processed_images = []
        
        for i, result in enumerate(results):
            # Load the result image with PIL
            img = Image.open(result.path)
            
            # Convert the image to RGB mode if it has an alpha channel
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # Process each bounding box
            for j, box in enumerate(result.boxes.xyxy):
                class_idx = int(result.boxes.cls[j].item())
                class_name = result.names[class_idx]
                confidence = result.boxes.conf[j].item()
                confidence = round(confidence, 2)
                
                # Extract document type, side, and orientation from the class name
                parts = class_name.split('_')
                
                # Extract and validate box coordinates
                xmin, ymin, xmax, ymax = map(float, box)
                
                # Ensure coordinates are valid
                if xmin >= xmax or ymin >= ymax:
                    continue
                    
                # Convert to integers for cropping
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                
                if len(parts) == 3:
                    doc_type, side, orient = parts
                    # Crop image - CRITICAL: Make a copy of the image first
                    img_copy = img.copy()
                    try:
                        # Ensure coordinates are within image bounds
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(img.width, xmax)
                        ymax = min(img.height, ymax)
                        
                        # Ensure box has minimum size
                        if xmax - xmin < 10 or ymax - ymin < 10:
                            continue
                            
                        cropped_img = img_copy.crop((xmin, ymin, xmax, ymax))
                    except Exception:
                        continue
                        
                    # Save cropped image with UUID to prevent overwrites
                    # but don't include UUID in filename if not needed for single documents
                    if page_info:
                        # Multi-page document - include unique identifiers
                        cropped_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_cropped.jpg'
                    else:
                        # Single document - keep simple name
                        cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_cropped.jpg'
                        
                    cropped_img_path = os.path.join(cropped_dir, cropped_img_name)
                    cropped_img.save(cropped_img_path)
                    processed_images.append(cropped_img_path)
                    
                    # Handle rotation
                    if orient in rotation_map:
                        rotation_angle = rotation_map[orient]
                        if rotation_angle != 0:
                            cropped_img = cropped_img.rotate(rotation_angle, expand=True)
                    
                    # Save oriented image with the same naming convention
                    if page_info:
                        oriented_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_oriented.jpg'
                    else:
                        oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_oriented.jpg'
                        
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    cropped_img.save(oriented_img_path)
                    processed_images.append(oriented_img_path)
                    
                else:
                    doc_type, orient = parts[0], parts[1]
                    side = 'front'  # No side information for certificates
                    
                    # For certificates, just save the whole image
                    img_copy = img.copy()
                    
                    # Use the same naming convention as above
                    if page_info:
                        non_cropped_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_non_cropped.jpg'
                    else:
                        non_cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_non_cropped.jpg'
                        
                    non_cropped_img_path = os.path.join(cropped_dir, non_cropped_img_name)
                    img_copy.save(non_cropped_img_path)
                    processed_images.append(non_cropped_img_path)
                    
                    # Save oriented copy
                    if page_info:
                        oriented_img_name = f'{doc_type}_{side}_{orient}{page_info}_{doc_uuid}_{i}_{j}_{confidence}_oriented.jpg'
                    else:
                        oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_oriented.jpg'
                        
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    img_copy.save(oriented_img_path)
                    processed_images.append(oriented_img_path)

        return processed_images

    processed_files = []
    if file_path.endswith('.pdf'):
        image_paths = process_pdf(file_path)
        for img_path in image_paths:
            processed_files.extend(process_image(img_path))
    else:
        processed_files.extend(process_image(file_path))

    return processed_files




# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
    