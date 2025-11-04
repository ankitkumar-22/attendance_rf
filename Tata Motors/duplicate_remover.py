import os
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
import cv2
import time

IMAGE_FOLDER = 'NoMaskImages'
LOG_FILE = 'log.xlsx'
THRESHOLD = 0.6  # ArcFace similarity threshold


def wait_for_file_access(filepath, mode='rb', delay=3):
    while True:
        try:
            with open(filepath, mode):
                print('File accessed.')
                return
        except PermissionError:
            print(f"Waiting for access to '{filepath}'... (is it open elsewhere?)")
            time.sleep(delay)


def extract_embedding(img_path):
    try:
        embedding_obj = DeepFace.represent(
            img_path=img_path,
            model_name='ArcFace',
            detector_backend='retinaface',
            enforce_detection=False
        )
        return embedding_obj[0]["embedding"]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def cosine_similarity(vec1, vec2):
    from numpy import dot
    from numpy.linalg import norm
    if vec1 is None or vec2 is None:
        return 0
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def main():
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return

    wait_for_file_access(LOG_FILE, 'rb')
    df = pd.read_excel(LOG_FILE, engine='openpyxl')
    image_files = df['Image Name'].unique()

    embeddings = []
    image_map = {}  # maps current image name to final kept image name

    print("Analyzing and comparing face embeddings...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not os.path.exists(img_path):
            continue

        embedding = extract_embedding(img_path)
        if embedding is None:
            continue

        duplicate_found = False
        for i, existing_embedding in enumerate(embeddings):
            sim = cosine_similarity(existing_embedding, embedding)
            if sim > THRESHOLD:
                kept_img = list(image_map.values())[i]
                print(f"Duplicate: {img_name} -> {kept_img} (similarity={sim:.2f})")
                image_map[img_name] = kept_img
                os.remove(img_path)
                duplicate_found = True
                break

        if not duplicate_found:
            embeddings.append(embedding)
            image_map[img_name] = img_name

    # Update log.xlsx
    df['Image Name'] = df['Image Name'].apply(lambda x: image_map.get(x, x))
    df.to_excel(LOG_FILE, index=False, engine='openpyxl')
    print("Duplicate removal and log update complete.")


if __name__ == "__main__":
    main()
