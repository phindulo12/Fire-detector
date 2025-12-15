from ultralytics import YOLO


# Paths

DATA_YAML = "C:/Users/rtsph/Downloads/noisy_wildfire_detector/noisy_wildfire_detector/dataset/data.yaml"       # points to your dataset
PRETRAINED_MODEL = "yolov8n.pt"  # lightweight pre-trained YOLOv8


# Initialize YOLOv8

model = YOLO(PRETRAINED_MODEL)


# Train the model

model.train(
    data=DATA_YAML,    # dataset config
    epochs=50,         # adjust for better accuracy
    imgsz=512,         # input image size
    batch=16,          # batch size
    name="fire_detection_new_2"  # output folder name
)


