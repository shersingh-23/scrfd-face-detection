# 🚀 SCRFD Face Detection (Image & Video)

<p align="center">
  <b>Fast & Accurate Face Detection using SCRFD (ONNX + OpenCV)</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue">
  <img src="https://img.shields.io/badge/OpenCV-4.x-orange">
  <img src="https://img.shields.io/badge/ONNX-Runtime-green">
</p>

---

## 🎥 Demo

### 🔹 GIF Preview

<p align="center">
  <img src="[output/demo.gif](https://github.com/shersingh-23/scrfd-face-detection/blob/main/outputs/demo.gif)" width="700">
</p>

---

### 🔹 Image Output

<p align="center">
  <img src="[output/output_result.jpg](https://github.com/shersingh-23/scrfd-face-detection/blob/main/outputs/output_result.jpg)" width="700">
</p>

---

## ✨ Features

* ⚡ Real-time Face Detection
* 🧠 SCRFD ONNX model (InsightFace)
* 🎯 High accuracy with multi-scale anchors
* 🎨 Confidence-based bounding box colors
* 👁️ Facial keypoints detection
* 🎥 Works on both Image & Video
* 🔧 Custom Non-Max Suppression (NMS)
* 📦 Adaptive padding for better face coverage
* 💻 CPU & GPU support (ONNX Runtime)

---

## 🧱 Project Structure

```
.
├── models/
│   └── det_10g.onnx
├── output/
│   ├── demo.gif
│   ├── output_result.jpg
│   └── output_video.mp4
├── scrfd_wrapper.py
├── test_image.py
├── test_video.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/shersingh-23/scrfd-face-detection.git
cd scrfd-face-detection

pip install -r requirements.txt
```

---

## 🖼️ Run on Image

```bash
python test_image.py
```

📌 Output:

```
output/output_result.jpg
```

---

## 🎥 Run on Video

```bash
python test_video.py
```

📌 Output:

```
output/output_video.mp4
```

---

## 🧠 Model Details

* Model: `det_10g.onnx`
* Framework: SCRFD (InsightFace)
* Backend: ONNX Runtime
* Input Size: 640×640
* Multi-scale anchors (stride 8, 16, 32)

---

## 🎨 Confidence Visualization

| Confidence | Color     |
| ---------- | --------- |
| ≥ 90%      | 🟢 Green  |
| 70–90%     | 🟠 Orange |
| < 70%      | 🔴 Red    |

---

## 🔍 How It Works

1. Preprocess image/frame
2. Run ONNX model inference
3. Decode bounding boxes & keypoints
4. Apply Non-Max Suppression (NMS)
5. Adaptive padding
6. Draw results

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* ONNX Runtime

---

## 🚀 Future Improvements

* 🔁 Face tracking (SORT / DeepSORT)
* 🧍 Face recognition
* 🌐 Web app (Streamlit / Flask)
* ⚡ TensorRT optimization

---

## 🙌 Acknowledgment

SCRFD model from InsightFace

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
