import cv2
from ultralytics import YOLO
from img_save_2 import save_image_and_log, close_connection

# Load YOLOv12 model trained for mask detection
model = YOLO("last.pt")  # Replace with your actual model path
print(model.names)
class_names = model.names
# Start webcam
cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, conf=0.5, verbose=False)

    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label_idx = int(cls_id)
            print(f"Predicted class ID: {cls_id}")
            confidence = float(conf)
            label = class_names[label_idx]

            # Draw bounding box and label
            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save image if no mask and high confidence
            face_img = frame[y1:y2, x1:x2]
            if label == "without_mask" and confidence > 0.9 and counter % 5 == 0:
                counter += 1
                save_image_and_log(face_img)

            if label == "WithMask":
                counter = 0

    cv2.imshow("YOLO Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_connection()
