import os
import cv2
import pandas as pd
import sqlite3
from datetime import datetime

# Folder where images will be saved
SAVE_FOLDER = 'NoMaskImages'
DB_FILE = 'log.db'  # SQLite database file

# Ensure the save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Create/connect to SQLite DB
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        date TEXT,
        time TEXT,
        location TEXT
    )
''')
conn.commit()


def save_image_and_log(image):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    filename = f"{date_str}_{time_str}.jpg"
    image_path = os.path.join(SAVE_FOLDER, filename)

    # Save image
    cv2.imwrite(image_path, image)

    location = "Engine Assembly"

    try:
        cursor.execute('''
            INSERT INTO logs (image_name, date, time, location)
            VALUES (?, ?, ?, ?)
        ''', (filename, date_str, time_str.replace('-', ':'), location))
        conn.commit()
        print(f"✅ Logged: {filename} at {date_str} {time_str}")
    except Exception as e:
        print(f"❌ Failed to log image: {e}")


def close_connection():
    conn.close()
    print("🔒 Database connection closed.")


# Call conn.close() when you're done
