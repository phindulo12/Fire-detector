import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.fire_ops import z9_execute_fire_thing
from ..modules.img_loader import loAd_fiLe


#  Load YOLOv8 Model (replace with your trained weights)

MODEL_PATH = "C:/Users/rtsph/Downloads/noisy_wildfire_detector/noisy_wildfire_detector/runs/detect/fire_detection_new/weights/best.pt"
# MODEL_PATH = "C:/Users/rtsph/Downloads/noisy_wildfire_detector/noisy_wildfire_detector/yolov8n.pt"
model = YOLO(MODEL_PATH)


#  Fire classification function using YOLOv8

def classify_fire(crop):
    """
    Takes a cropped image (candidate region) and returns:
        is_fire: bool
        confidence: float
    """
    results = model(crop)
    boxes = results[0].boxes  # YOLOv8 Box objects
    if boxes is not None and len(boxes) > 0:
        # Take the highest confidence score in the crop
        confidence = max(boxes.conf.tolist())
        return True, float(confidence)
    else:
        return False, 0.0


#  Main Processing Pipeline

def main():
    # Load image
    path = "img1.png"
    img = loAd_fiLe(path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")

    # Get raw detections (candidate bounding boxes)
    detections = z9_execute_fire_thing(img)

    # Store YOLO-confirmed detections
    filtered_detections = []

    for d in detections:
        x, y, w, h = d["bx"], d["by"], d["bw"], d["bh"]

        # Crop candidate region safely
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        # Use YOLO to confirm fire
        is_fire, confidence = classify_fire(crop)

        if is_fire:
            d["yolo_conf"] = confidence
            filtered_detections.append(d)

            # Draw confirmed detection (red)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f"FIRE {confidence:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Optional: draw rejected boxes (blue)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(img, f"NO {confidence:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Show final image
    cv2.imshow("Fire Detection YOLOv8", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Detections:", filtered_detections)


#  Entrypoint

if __name__ == "__main__":
    main()
