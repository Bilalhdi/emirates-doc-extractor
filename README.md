# Emirates Document Extractor API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)  
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)  
![Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash_Lite-yellow)

A high-performance REST API for extracting structured data from **UAE identity & vehicle documents** using **YOLOv8 (detect/crop/orient)** and **Google Gemini 2.5 Flash Lite**.

## Table of Contents
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  


## Features
- **YOLOv8 Detection** – Classifies doc type (ID/Driving/Mulkiya/Trade/Pass), side (front/back), and orientation.  
- **Auto Crop & Rotate** – Crops detected regions and fixes orientation before extraction.  
- **LLM Extraction** – Prompts Gemini to return **strict JSON** (no prose/markdown).  
- **Multi-Format Input** – `.jpg`, `.png`, `.jfif`, `.heic/.heif`, and multi-page `.pdf`.  
- **Two Ingestion Modes** – Multipart file upload or Base64 JSON payloads.  
- **Secure Config** – Reads keys/paths from `.env` (never hard-code secrets).  

## Prerequisites

### System Requirements
- **Python 3.9+**  
- **Google Gemini API key** (create at https://aistudio.google.com)  
- **YOLOv8 trained weights** at `models/yolov8n-emirates.pt`  

### Python Packages  
Listed in `requirements.txt`:
```python
fastapi
uvicorn
ultralytics
opencv-python
pillow
pillow-heif
pymupdf
numpy
easyocr
python-dotenv
google-generativeai
```
##Installation

1. **Clone the repository**  
   ```python
   git clone https://github.com/<your-username>/emirates-doc-extractor.git
   cd emirates-doc-extractor
   ```
2. **Create & activate Conda env**
   ```python
   conda create -n docreader python=3.10 -y
   conda activate docreader
   ```
3. **Install Dependancies**
   ```python
   pip install -r requirements.txt
   ```
###Configuration

1. Create a .env file in the project root
   ```python
   GEMINI_API_KEY=your_company_gemini_api_key_here
   YOLO_WEIGHTS_PATH=models/yolov8n-emirates.pt
   PORT=4000
   ```
2. Code already loads env (via python-dotenv)

###API Usage

##Start the server
```python
  


