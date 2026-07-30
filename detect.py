"""
Real-Time Object Detection using YOLOv8 and OpenCV
Author: Badal Mishra

This script performs object detection on images (or webcam feed) using
the YOLOv8 model from Ultralytics, combined with OpenCV for image handling
and visualization.
"""

import cv2
from ultralytics import YOLO


def load_model(model_path="yolov8n.pt"):
    """Load the pretrained YOLOv8 nano model."""
    model = YOLO(model_path)
    return model


def detect_on_image(model, image_path, output_path="output.jpg"):
    """Run object detection on a single image and save the annotated result."""
    results = model(image_path)

    for result in results:
        annotated_frame = result.plot()  # Draws bounding boxes + labels
        cv2.imwrite(output_path, annotated_frame)

        print(f"\nDetected {len(result.boxes)} objects:")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"  - {label}: {conf * 100:.1f}% confidence")

    print(f"\nAnnotated image saved to: {output_path}")
    return results


def detect_on_webcam(model):
    """Run real-time object detection using your webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Press 'q' to quit webcam detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Loading YOLOv8 model...")
    model = load_model()

    print("\nRunning detection on sample image...")
    detect_on_image(model, "sample.jpg", "output.jpg")

    # Uncomment the line below to try real-time webcam detection
    # detect_on_webcam(model)


if __name__ == "__main__":
    main()
