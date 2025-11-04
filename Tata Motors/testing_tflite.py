import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mask_detection_model2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape (height, width)
input_shape = input_details[0]['shape'][1:3]
input_dtype = input_details[0]['dtype']

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

class_names = ['WithMask', 'WithoutMask']

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Model input shape: {input_shape}")
print(f"Class names: {class_names}")

# --- Counter for 'WithoutMask' ---
without_mask_counter = 0
REQUIRED_WITHOUT_MASK_COUNT = 5  # Adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
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

            # Clip coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop face and preprocess
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            face_resized = cv2.resize(face_img, (input_shape[1], input_shape[0]))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0).astype(input_dtype)

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], face_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get prediction
            label_index = np.argmax(output)
            current_label = class_names[label_index]
            confidence = output[label_index]

            # Counter Logic
            display_label = ""
            display_color = (0, 255, 0)  # Default green

            if current_label == "WithoutMask":
                without_mask_counter += 1
                if without_mask_counter > REQUIRED_WITHOUT_MASK_COUNT:
                    display_label = f"WithoutMask ({confidence * 100:.2f}%)"
                    display_color = (0, 0, 255)  # Red
                else:
                    display_label = f"WithMask (Counting... {without_mask_counter}/{REQUIRED_WITHOUT_MASK_COUNT})"
            else:
                without_mask_counter = 0
                display_label = f"WithMask ({confidence * 100:.2f}%)"

            # Draw label and bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

    # Show the frame
    cv2.imshow("Mask Detection - TensorFlow Lite Model", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
