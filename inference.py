import os
from ultralytics import YOLO
import cv2

# YOLOv8 model path
model_path = r"runs\idcard_yolov8n9\weights\best.pt"
# Validation images directory
val_dir = r"C:\Users\bilal\OneDrive\Desktop\Agile\Emirates stuff\annotateID\images\val"
# Output root directory
output_root = r"C:\Users\bilal\OneDrive\Desktop\Agile\Emirates stuff\annotateID\cropped outputs test"

# CLASS NAMES (index must match model training)
CLASS_NAMES = [
    "ID_front_0", "ID_front_90", "ID_front_180", "ID_front_270",
    "ID_back_0", "ID_back_90", "ID_back_180", "ID_back_270",
    "vehicle_front_0", "vehicle_front_90", "vehicle_front_180", "vehicle_front_270",
    "vehicle_back_0", "vehicle_back_90", "vehicle_back_180", "vehicle_back_270",
    "pass_front_0", "pass_front_90", "pass_front_180", "pass_front_270",
    "trade_front_0", "trade_front_90", "trade_front_180", "trade_front_270",
    "Driving_front_0", "Driving_front_90", "Driving_front_180", "Driving_front_270",
    "Driving_back_0", "Driving_back_90", "Driving_back_180", "Driving_back_270"
]

model = YOLO(model_path)

# Supported image extensions
exts = {".jpg", ".jpeg", ".png"}

for fname in os.listdir(val_dir):
    if not os.path.splitext(fname)[1].lower() in exts:
        continue
    img_path = os.path.join(val_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: couldn't read image {img_path}")
        continue

    # Run inference
    results = model(img, conf=0.3)  # Set conf threshold as needed

    # Loop through detections
    for i, r in enumerate(results[0].boxes):
        cls_id = int(r.cls[0])
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[cls_id]

        # Bounding box coordinates (xyxy format, float to int)
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        crop = img[y1:y2, x1:x2]

        # Output path: ...\class_name\originalfilename_idx.jpg
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{base_name}_{i+1}.jpg")

        cv2.imwrite(out_path, crop)
        print(f"Cropped object saved to {out_path}")
