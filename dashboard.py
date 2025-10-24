import os
import torch
import streamlit as st
from ultralytics import YOLO

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_yolo_model(model_path):
    return YOLO(model_path)

uploaded_file = st.file_uploader("Upload model YOLO (.pt)", type=["pt"])

if uploaded_file:
    temp_path = "temp_model.pt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        yolo_model = load_yolo_model(temp_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    
    # Hapus file temporary jika perlu
    os.remove(temp_path)
