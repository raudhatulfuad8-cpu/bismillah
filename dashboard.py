import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# Load model YOLOv8 dari file .pt
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)
    return model

# Fungsi menampilkan bounding box dan label pada gambar hasil deteksi
def draw_boxes(img, results):
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = results.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

# Main streamlit app
def main():
    st.title("Deteksi Hewan (Cheetah atau Lion) dengan YOLOv8 dan Streamlit")

    model_path = "yolov8_cheetah_lion.pt"  # Ganti dengan path model anda
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload gambar hewan untuk deteksi", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar asli", use_column_width=True)

        # Konversi PIL image ke numpy untuk cv2
        img = np.array(image)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Inferensi menggunakan YOLOv8
        results = model(img_cv)

        # Gambarkan bounding box dan label di gambar
        img_result = draw_boxes(img_cv, results)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        st.image(img_result, caption="Hasil Deteksi", use_column_width=True)
        st.write("Deteksi berhasil dilakukan!")

if __name__ == "__main__":
    main()
