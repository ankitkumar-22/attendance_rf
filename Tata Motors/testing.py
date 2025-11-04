import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from img_save_2 import save_image_and_log
from img_save_2 import close_connection



# Load Keras model
model = tf.keras.models.load_model('mask_detection_model_4.keras')
class_names = ['WithMask', 'WithoutMask']

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
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + w_box) * w)
            y2 = int((y + h_box) * h)

            # Crop face and preprocess
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_resized = cv2.resize(face_img, (224, 224))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            # Predict
            preds = model.predict(face_input)[0]
            label_idx = np.argmax(preds)
            label = class_names[label_idx]
            confidence = preds[label_idx]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if label == "WithMask" else (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if label == "WithoutMask" and confidence > 0.9 and counter % 3 == 0:
                counter = counter + 1
                save_image_and_log(face_img)

            if label == "WithMask":
                counter = 0

    cv2.imshow("Mask Detection with Face Cropping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_connection()
