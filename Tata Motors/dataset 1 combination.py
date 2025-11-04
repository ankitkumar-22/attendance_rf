import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from skimage.metrics import brisque_score  # You'll need scikit-image

input_img_dir = "images"
input_xml_dir = "annotations"
output_base = "dataset1_cropped_faces"

TARGET_SIZE = (224, 224)

# BRISQUE threshold (lower score is better quality)
# This value needs careful tuning for your dataset.
BRISQUE_QUALITY_THRESHOLD = 50.0  # Experiment with this value!

os.makedirs(os.path.join(output_base, "with_mask"), exist_ok=True)
os.makedirs(os.path.join(output_base, "without_mask"), exist_ok=True)

processed_count = 0
discarded_blur_count = 0
discarded_empty_face_count = 0
discarded_img_not_found_count = 0

for xml_file in tqdm(os.listdir(input_xml_dir)):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_xml_dir, xml_file))
    root = tree.getroot()
    img_file = root.find("filename").text
    img_path = os.path.join(input_img_dir, img_file)

    if not os.path.exists(img_path):
        discarded_img_not_found_count += 1
        continue

    img = cv2.imread(img_path)

    if img is None:
        # print(f"Warning: Could not read image {img_path}. Skipping.")
        discarded_img_not_found_count += 1
        continue

    for i, obj in enumerate(root.findall("object")):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        h, w, _ = img.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        face = img[ymin:ymax, xmin:xmax]

        if face.shape[0] == 0 or face.shape[1] == 0:
            discarded_empty_face_count += 1
            continue

        # Convert the face to grayscale for BRISQUE calculation
        # BRISQUE is typically applied to grayscale images
        gray_face_for_brisque = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Check if the image is too small for BRISQUE (it needs some features)
        # BRISQUE works best on images with reasonable size (e.g., > 32x32 pixels)
        if gray_face_for_brisque.shape[0] < 32 or gray_face_for_brisque.shape[1] < 32:
            # print(f"Warning: Face too small for reliable BRISQUE score from {img_file}, object {i}. Cropped size: {gray_face_for_brisque.shape[:2]}. Skipping.")
            discarded_blur_count += 1  # Counting as blurry/low quality due to size
            continue

        # Calculate BRISQUE score
        # Note: brisque_score expects a float64 image
        brisque = brisque_score(gray_face_for_brisque.astype(np.float64))

        # Discard if the BRISQUE score indicates low quality (higher score = worse quality)
        if brisque > BRISQUE_QUALITY_THRESHOLD:
            # print(f"Discarding low quality face from {img_file}, object {i}. BRISQUE score: {brisque:.2f}")
            discarded_blur_count += 1
            continue

        # Resize the face to the target size *after* quality check
        face_resized = cv2.resize(face, TARGET_SIZE)

        label_dir = os.path.join(output_base, label)
        os.makedirs(label_dir, exist_ok=True)

        face_filename = f"{os.path.splitext(img_file)[0]}_{i}.jpg"

        cv2.imwrite(os.path.join(label_dir, face_filename), face_resized)
        processed_count += 1

print(f"Processing complete!")
print(f"Total images processed and saved: {processed_count}")
print(f"Images discarded due to low quality (BRISQUE): {discarded_blur_count}")
print(f"Faces discarded due to empty crop (invalid bounding box): {discarded_empty_face_count}")
print(f"Images skipped because source file not found: {discarded_img_not_found_count}")