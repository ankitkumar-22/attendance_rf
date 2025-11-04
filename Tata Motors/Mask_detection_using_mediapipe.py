import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1)

# Landmark indices for areas typically covered by a mask
# (nose, mouth, and surrounding areas)
MASK_RELEVANT_LANDMARKS = [
    # Nose and bridge
    48, 120, 115, 220, 45, 4, 275, 440, 344, 278, 279, 294, 438, 439,
    216, 92, 165, 167, 164, 393, 391, 322, 410, 287, 432,
    302, 303, 304, 310, 311, 312, 313, 0, 11, 12, 13, 37, 72, 38, 82, 39, 73, 41, 81,
    57, 43, 91, 181, 84, 17, 314, 405, 321, 375,
    335, 406, 313, 18, 83, 182, 106,
    202, 204, 194, 201, 200, 421, 418, 424, 422, 432,
    430, 431, 262, 428, 199, 208, 32, 211, 210, 214,
    #192, 215, 58, 138, 172, 135, 138, 169, 170, 150, 140, 149, 171, 176, 148, 175, 152, 396, 377, 369, 400, 378, 379,
    #394, 365, 431, 397, 364, 430, 367, 288, 434, 416, 435, 361, 433, 401, 411, 378

]

# Thresholds
VISIBILITY_THRESHOLD = 0  # Z-value threshold for visibility
MIN_VISIBLE_POINTS = len(MASK_RELEVANT_LANDMARKS)*0.8  # If more than this many points are visible → no mask


def get_visibility_percentage(face_landmarks):
    """Calculate percentage of mask-relevant landmarks that are visible"""
    visible_count = 0

    for landmark_idx in MASK_RELEVANT_LANDMARKS:
        landmark = face_landmarks.landmark[landmark_idx]
        if landmark.z < VISIBILITY_THRESHOLD:  # More negative = more visible
            visible_count += 1

    total_points = len(MASK_RELEVANT_LANDMARKS)
    return (visible_count / total_points) * 100, visible_count


# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate visibility
            visibility_percent, visible_points = get_visibility_percentage(face_landmarks)
            print(len(face_landmarks.landmark))
            # Determine mask status
            if visible_points > MIN_VISIBLE_POINTS:
                mask_status = "NO MASK DETECTED"
                color = (0, 0, 255)  # Red
            else:
                mask_status = "MASK DETECTED"
                color = (0, 255, 0)  # Green

            # Display the results
            cv2.putText(image, mask_status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Visible points: {visible_points}/{len(MASK_RELEVANT_LANDMARKS)}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Visibility: {visibility_percent:.1f}%",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Optional: Draw the relevant landmarks
            for landmark_idx in MASK_RELEVANT_LANDMARKS:
                landmark = face_landmarks.landmark[landmark_idx]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    # Show the image
    cv2.imshow('Enhanced Mask Detection', image)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
