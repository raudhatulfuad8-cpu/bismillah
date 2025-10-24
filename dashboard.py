import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from tensorflow.keras.models import load_model

# Load model YOLOv8 untuk deteksi objek
@st.cache(allow_output_mutation=True)
def load_yolo_model(path):
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=path, force_reload=True)
    return model

# Load model Keras (.h5) untuk klasifikasi
@st.cache(allow_output_mutation=True)
def load_classifier_model(path):
    model = load_model(path)
    return model

# Preprocess gambar untuk model klasifikasi
def preprocess_classifier_img(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Fungsi gambar bounding box deteksi YOLOv8
def draw_boxes(img, results):
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = results.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return img

def main():
    st.set_page_config(page_title="Deteksi & Klasifikasi Hewan dengan YOLOv8 dan Keras", layout="wide")
    st.title("🌟 Deteksi dan Klasifikasi Hewan: Cheetah & Lion")
    st.markdown("Upload gambar dan aplikasi akan menampilkan hasil deteksi dan klasifikasi.")

    with st.sidebar:
        st.header("Pengaturan Model")
        yolo_model_path = st.text_input("Path model YOLOv8 (.pt)", "best.pt")
        classifier_model_path = st.text_input("Path model klasifikasi (.h5)", "classifier_model.h5")
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

    yolo_model = load_yolo_model(yolo_model_path)
    classifier_model = load_classifier_model(classifier_model_path)

    uploaded_file = st.file_uploader("Upload gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar Asli", use_column_width=True)

        # Deteksi menggunakan YOLOv8
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = yolo_model(img_cv, conf=confidence_threshold)
        img_boxed = draw_boxes(img_cv.copy(), results)
        img_boxed_rgb = cv2.cvtColor(img_boxed, cv2.COLOR_BGR2RGB)

        st.subheader("Hasil Deteksi YOLOv8")
        st.image(img_boxed_rgb, use_column_width=True)

        # Klasifikasi gambar menggunakan model .h5
        preprocessed_img = preprocess_classifier_img(image)
        classifier_preds = classifier_model.predict(preprocessed_img)
        class_idx = np.argmax(classifier_preds)
        confidence = classifier_preds[0][class_idx]

        class_labels = ['Cheetah', 'Lion']  # Sesuaikan dengan label model klasifikasi Anda
        predicted_class = class_labels[class_idx]

        st.subheader("Hasil Klasifikasi Gambar")
        st.write(f"Prediksi: **{predicted_class}** dengan confidence **{confidence:.2f}**")

        st.success("Deteksi dan klasifikasi selesai!")

if __name__ == "__main__":
    main()
