import os
import cv2
import pandas as pd
from datetime import datetime


# Folder where images will be saved
SAVE_FOLDER = 'NoMaskImages'
LOG_FILE = 'log.xlsx'

# Ensure the save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)


def save_image_and_log(image):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    filename = f"{date_str}_{time_str}.jpg"
    image_path = os.path.join(SAVE_FOLDER, filename)

    # Save the image
    cv2.imwrite(image_path, image)

    # Get location (static or dynamic if available)
    location = "Engine Assembly"

    # Prepare log entry
    log_data = {
        'Image Name': filename,
        'Date': date_str,
        'Time': time_str.replace('-', ':'),
        'Location': location
    }

    # Append or create the Excel file
    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)

    else:
        df = pd.DataFrame([log_data])

    df.to_excel(LOG_FILE, index=False)
