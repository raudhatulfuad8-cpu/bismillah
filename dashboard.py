import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tempfile
import os

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Lions vs Cheetahs Detector 🦁🐆",
    page_icon="🦁",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem;}
        h1, h2, h3 {text-align: center;}
        .stImage {border-radius: 12px;}
    </style>
""", unsafe_allow_html=True)

st.title("🦁🐆 Deteksi & Klasifikasi Hewan: Lions vs Cheetahs")
st.write("Aplikasi ini mendeteksi objek dengan **YOLOv8** dan mengklasifikasikan gambar apakah itu **Singa (Lion)** atau **Cheetah** menggunakan model .h5 milikmu.")

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_models():
    yolo = YOLO("model/best.pt")  # model YOLOv8 deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # model klasifikasi (Lion vs Cheetah)
    return yolo, classifier

yolo_model, classifier_model = load_models()

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar hewan (lion atau cheetah)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan sementara file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    st.image(uploaded_file, caption="📸 Gambar yang diunggah", use_column_width=True)

    # ==========================
    # 1️⃣ Deteksi Objek (YOLOv8)
    # ==========================
    st.subheader("📦 Hasil Deteksi Objek (YOLOv8)")
    results = yolo_model(temp_file.name)
    detected_image = results[0].plot()  # hasil visualisasi YOLO
    st.image(detected_image, caption="Hasil Deteksi YOLOv8", use_column_width=True)

    # ==========================
    # 2️⃣ Klasifikasi (Model .h5)
    # ==========================
    st.subheader("🔎 Hasil Klasifikasi (Model .h5)")

    # Pra-pemrosesan untuk model .h5 kamu
    img = Image.open(temp_file.name).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = classifier_model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    probability = np.max(pred)

    # Label klasifikasi (ubah sesuai label model kamu)
    class_labels = ["Cheetah", "Lion"]

    result_label = class_labels[class_idx]
    emoji = "🐆" if result_label == "Cheetah" else "🦁"

    st.success(f"Hasil prediksi: **{result_label} {emoji}**")
    st.caption(f"Probabilitas prediksi: **{probability:.4f}**")

    temp_file.close()

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Dibuat oleh Raudah 🩵 | Menggunakan YOLOv8 & TensorFlow | Aplikasi deteksi otomatis Lions vs Cheetahs")
