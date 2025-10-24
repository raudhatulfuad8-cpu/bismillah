import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from tensorflow.keras.models import load_model

# Styling CSS untuk tema warna dan font
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #e0f2f1;
        color: #00796b;
        font-weight: bold;
    }
    .stFileUploader>div>div {
        border: 2px dashed #4caf50;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache(allow_output_mutation=True)
def load_yolo_model(path):
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=path, force_reload=True)
    return model

@st.cache(allow_output_mutation=True)
def load_classifier_model(path):
    model = load_model(path)
    return model

def preprocess_classifier_img(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def draw_boxes(img, results):
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = results.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 0), 3)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0), 2)
    return img

def main():
    st.set_page_config(page_title="Deteksi dan Klasifikasi Hewan", layout="wide")

    st.title("🦁 Deteksi dan Klasifikasi Hewan: Cheetah & Lion")
    st.write("Upload gambar hewan dan aplikasinya akan mendeteksi objek dengan YOLOv8 dan mengklasifikasikan gambar menggunakan model Keras.")

    with st.sidebar:
        st.header("Pengaturan Model")
        yolo_model_path = st.text_input("Path model YOLOv8 (.pt)", "best.pt")
        classifier_model_path = st.text_input("Path model klasifikasi (.h5)", "classifier_model.h5")
        confidence_threshold = st.slider("Confidence Threshold YOLOv8", 0.0, 1.0, 0.25, 0.05)
        st.markdown("---")
        st.markdown("**Catatan:** Pastikan model berada di lokasi yang benar dan kompatibel.")

    try:
        yolo_model = load_yolo_model(yolo_model_path)
        classifier_model = load_classifier_model(classifier_model_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    uploaded_file = st.file_uploader("Upload Gambar Hewan (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar asli", use_column_width=True)

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = yolo_model(img_cv, conf=confidence_threshold)

        img_result = draw_boxes(img_cv.copy(), results)
        img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        st.subheader("Hasil Deteksi dengan YOLOv8")
        st.image(img_result_rgb, use_column_width=True)

        preprocessed = preprocess_classifier_img(image)
        preds = classifier_model.predict(preprocessed)
        class_idx = np.argmax(preds)
        class_labels = ['Cheetah', 'Lion']  # Sesuaikan sesuai model klasifikasi Anda
        conf_score = preds[0][class_idx]
        predicted_class = class_labels[class_idx]

        st.subheader("Hasil Klasifikasi Gambar")
        st.markdown(f"""
            <h3 style="color:#4caf50;">{predicted_class}</h3>
            <p>Confidence: <strong>{conf_score:.2f}</strong></p>
        """, unsafe_allow_html=True)

        st.success("Deteksi dan klasifikasi selesai!")

if __name__ == "__main__":
    main()
