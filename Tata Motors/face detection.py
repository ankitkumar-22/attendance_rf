import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bbox = detection.location_data.relative_bounding_box

                # Get image dimensions
                height, width, _ = image.shape

                # Convert relative coordinates to absolute pixel values
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Print coordinates (x, y, width, height)
                print(f"Face at (x={x}, y={y}), width={w}, height={h}")

                # Draw bounding box (optional)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Face Detection with Coordinates', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()