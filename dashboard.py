import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ==============================
# 1️⃣ KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Klasifikasi & Deteksi Objek", page_icon="✨", layout="wide")
st.title("📸 Aplikasi Deteksi & Klasifikasi Gambar")
st.write("Unggah gambar untuk dideteksi oleh **YOLOv8 (.pt)** dan diklasifikasi oleh **Model H5**.")

# ==============================
# 2️⃣ LOAD MODEL (CACHE)
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    classifier = tf.keras.models.load_model("classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==============================
# 3️⃣ UPLOAD GAMBAR
# ==============================
uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Tombol proses
    if st.button("🚀 Jalankan Model"):
        with st.spinner("Model sedang menganalisis gambar..."):
            # ==============================
            # 4️⃣ KLASIFIKASI GAMBAR (.H5)
            # ==============================
            img_resized = image.resize((224, 224))  # ubah ukuran sesuai arsitektur model
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = classifier.predict(img_array)
            class_index = np.argmax(preds)
            confidence = np.max(preds)

            st.subheader("📦 Hasil Klasifikasi (.h5)")
            st.write(f"**Kelas Prediksi:** `{class_index}` | **Kepercayaan:** {confidence*100:.2f}%")

            # ==============================
            # 5️⃣ DETEKSI OBJEK (YOLOv8)
            # ==============================
            results = yolo_model.predict(np.array(image), conf=0.3)
            result_img = results[0].plot()  # hasil dengan bounding box

            st.subheader("🎯 Hasil Deteksi Objek (YOLOv8)")
            st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

            # Tampilkan detail deteksi
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.write("**Objek Terdeteksi:**")
                for i, box in enumerate(boxes):
                    cls_name = yolo_model.names[int(box.cls)]
                    conf_score = float(box.conf)
                    st.write(f"🔹 {i+1}. {cls_name} ({conf_score*100:.1f}%)")
            else:
                st.info("Tidak ada objek yang terdeteksi.")
