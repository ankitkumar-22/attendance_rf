import cv2
import numpy as np
import mediapipe as mp
from img_save_2 import save_image_and_log, close_connection
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.5  # optional: set confidence threshold

# Set class names (override if needed)
class_names = model.names  # e.g., {0: 'NoMask', 1: 'Mask'}

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box from MediaPipe
            bbox = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + w_box) * w)
            y2 = int((y + h_box) * h)

            # Crop face region
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Predict using YOLOv5 on cropped face
            results_yolo = model(face_img, size=224)  # resize for faster inference
            detections = results_yolo.pandas().xyxy[0]

            if not detections.empty:
                top_pred = detections.iloc[0]  # get highest confidence prediction
                label_idx = int(top_pred['class'])
                label = class_names[label_idx]
                confidence = top_pred['confidence']

                # Draw box and label
                color = (0, 255, 0) if label.lower() in ['withmask', 'mask'] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Logging if no mask and high confidence
                if label.lower() in ['withoutmask', 'nomask'] and confidence > 0.9 and counter % 3 == 0:
                    counter += 1
                    save_image_and_log(face_img)

                if label.lower() in ['withmask', 'mask']:
                    counter = 0

    cv2.imshow("Mask Detection with Face Cropping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_connection()
