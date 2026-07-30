#  Real-Time Object Detection using YOLOv8 and OpenCV

A Computer Vision project that detects and labels multiple objects in images (and optionally live webcam feed) using the **YOLOv8** deep learning model combined with **OpenCV** for visualization.

## 🚀 Features
- Detects multiple objects (people, vehicles, animals, everyday items, etc.) in a single image
- Draws bounding boxes with class labels and confidence scores
- Prints a clean summary of all detected objects in the terminal
- Optional real-time webcam detection mode
- Uses YOLOv8n (nano) — lightweight and fast, ideal for quick inference

## 🛠️ Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV

## Project Structure
```
object-detection/
│
├── detect.py           # Main script (image + webcam detection)
├── sample.jpg           # Sample input image for testing
├── output.jpg           # Output image with detected objects (bounding boxes)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/your-username/object-detection.git
cd object-detection
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

##  Usage

Run detection on the sample image:
```bash
python detect.py
```

This will:
1. Load the pretrained YOLOv8n model (auto-downloads on first run)
2. Run detection on `sample.jpg`
3. Print detected objects with confidence scores
4. Save the annotated result as `output.jpg`

### Try it on your own image
Replace `sample.jpg` in `detect.py` with the path to your own image.

### Real-time webcam detection
Uncomment the `detect_on_webcam(model)` line inside `main()` in `detect.py` to run live detection using your webcam.

##  How It Works
1. **Model Loading** – YOLOv8n, a pretrained object detection model, is loaded using the Ultralytics library.
2. **Inference** – The model processes an input image and predicts bounding boxes, class labels, and confidence scores for detected objects.
3. **Visualization** – OpenCV is used to draw the bounding boxes and labels directly on the image, which is then saved as output.
4. **Reporting** – Detected object names and their confidence percentages are printed to the console.

##  Sample Output
Input image with 6 objects detected: 4 persons, 1 bus, and 1 stop sign — each with bounding boxes and confidence scores.

##  Future Improvements
- Add support for video file input (not just webcam/images)
- Build a simple web UI using Streamlit to upload and test images
- Fine-tune YOLOv8 on a custom dataset for a specific use case (e.g., helmet detection, license plate detection)
- Deploy as a REST API using FastAPI

## 👤 Author
**Badal Mishra**
B.Tech CSE (Artificial Intelligence)
[LinkedIn](https://linkedin.com/in/badal-mishra-6430573ab)
