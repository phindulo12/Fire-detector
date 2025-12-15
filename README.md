# 🔥 Fire Detection System using YOLOv8

## 📌 Overview

This project is a **fire detection system** built using **YOLOv8 (You Only Look Once v8)**. The model is trained to detect fire in images and videos in real time, making it suitable for **early fire warning systems**, **surveillance**, and **safety monitoring applications**.

The system leverages deep learning and computer vision to accurately identify fire regions and draw bounding boxes around detected flames.

---

## 🚀 Features

* 🔍 Real-time fire detection
* 🧠 Powered by YOLOv8 (Ultralytics)
* 🎥 Supports images, videos, and webcam streams
* 📦 Easy to train on custom datasets
* ⚡ Fast inference speed

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **PyTorch**
* **NumPy**

---

## 📂 Project Structure

```
fire-detector-yolov8/
│── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│── runs/
│── data.yaml
│── train.py
│── detect.py
│── requirements.txt
│── README.md
```

---

## 📊 Dataset Format

The dataset follows the **YOLO format**:

```
<class_id> <x_center> <y_center> <width> <height>
```

* Coordinates are normalized between **0 and 1**
* Example class mapping:

```yaml
names:
  0: fire
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/fire-detector-yolov8.git
cd fire-detector-yolov8
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Training the Model

Edit `data.yaml` to point to your dataset paths, then run:

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

Trained models will be saved in:

```
runs/detect/train/
```

---

## 🔎 Running Detection

### Detect fire in an image

```bash
yolo detect model=best.pt source=image.jpg
```

### Detect fire in a video

```bash
yolo detect model=best.pt source=video.mp4
```

### Webcam detection

```bash
yolo detect model=best.pt source=0
```

---

## 📈 Results

* Bounding boxes drawn around detected fire regions
* Confidence scores displayed for each detection
* High accuracy with properly labeled datasets

---

## ⚠️ Limitations

* May struggle in **fire-like lighting conditions**
* Performance depends heavily on dataset quality
* Not a replacement for certified fire safety systems

---

## 🔮 Future Improvements

* Smoke detection integration
* Alarm & notification system
* Edge deployment (Raspberry Pi / Jetson Nano)
* Mobile and web-based dashboard

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Phindulo Ratshilumela**
Information Systems / AI Developer

---

## ⭐ Acknowledgements

* [Ultralytics YOLOv8]
* Open-source fire datasets
* PyTorch community

---

Feel free to ⭐ star the repository and contribute!
