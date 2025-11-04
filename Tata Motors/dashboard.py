import streamlit as st
import sqlite3
import pandas as pd
import os
from PIL import Image

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'log.db')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'NoMaskImages')

# Streamlit page config
st.set_page_config(page_title="No Mask Detection Dashboard", layout="wide")
st.title("📸 No Mask Detection Dashboard")

# Debug print
st.sidebar.success(f"DB Path: {DB_FILE}")
st.sidebar.success(f"Image Folder: {IMAGE_FOLDER}")

# Connect to DB
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Test query and display results
try:
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY date DESC, time DESC", conn)
    st.success(f"✅ Rows fetched from DB: {len(df)}")
    st.json(list(df.columns))
except Exception as e:
    st.error(f"❌ DB Read Error: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ No records found in the database.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
unique_dates = sorted(df['date'].unique(), reverse=True)
selected_date = st.sidebar.selectbox("Filter by Date", options=["All"] + unique_dates)

unique_locations = sorted(df['location'].unique())
selected_location = st.sidebar.selectbox("Filter by Location", options=["All"] + unique_locations)

# Apply filters
filtered_df = df.copy()
if selected_date != "All":
    filtered_df = filtered_df[filtered_df['date'] == selected_date]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['location'] == selected_location]

st.write(f"### Showing {len(filtered_df)} entries")
st.dataframe(filtered_df[['image_name', 'date', 'time', 'location']], use_container_width=True)

# Display images with metadata
st.write("### 📷 Images")
for _, row in filtered_df.iterrows():
    image_path = os.path.join(IMAGE_FOLDER, row['image_name'])
    if os.path.exists(image_path):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(Image.open(image_path), width=150, caption=row['image_name'])
        with col2:
            st.markdown(f"""
            - **📅 Date**: {row['date']}
            - **⏰ Time**: {row['time']}
            - **📍 Location**: {row['location']}
            """)
        st.markdown("---")
    else:
        st.warning(f"⚠️ Image not found: {image_path}")

conn.close()
