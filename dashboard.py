import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Lion vs Cheetah Detection", page_icon="🦁", layout="wide")

st.title("🦁 Lion vs Cheetah Object Detection and Classification")
st.markdown("Upload gambar untuk mendeteksi objek dengan YOLOv8 dan klasifikasi dengan model kamu (.h5).")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")  # model deteksi objek YOLOv8
    
    # 🔧 trik kompatibilitas: disable compile dan safe_mode
    classifier_model = tf.keras.models.load_model(
        "classifier_model.h5",
        compile=False
    )
    return yolo_model, classifier_model

yolo_model, classifier_model = load_models()

# ==========================
# UPLOAD FILE GAMBAR
# ==========================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    # Convert ke array
    img_array = np.array(image)

    # ==========================
    # DETEKSI DENGAN YOLOv8
    # ==========================
    st.subheader("🔍 Deteksi Objek dengan YOLOv8")
    results = yolo_model(img_array)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Hasil Deteksi YOLOv8", use_column_width=True)

    # ==========================
    # KLASIFIKASI GAMBAR
    # ==========================
    st.subheader("🧠 Klasifikasi Gambar (Lion vs Cheetah)")
    resized_img = image.resize((224, 224))  # sesuaikan dengan input model kamu
    img_batch = np.expand_dims(np.array(resized_img) / 255.0, axis=0)

    pred = classifier_model.predict(img_batch)
    kelas = ["Cheetah", "Lion"]
    hasil = kelas[np.argmax(pred)]

    st.success(f"Hasil klasifikasi: **{hasil}** 🐾")
