from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    
    model = YOLO('yolov8n.pt')  # or your own weights if resuming
    model.train(
        data='annotateID/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='idcard_yolov8n',
        project='runs',
        device=0,
    )

if __name__ == "__main__":
    main()