import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import io
import numpy as np

# --- KONSTANTA & SETUP AWAL ---
# File model yang diunggah
H5_FILE_PATH = "classifier_model.h5"
PT_FILE_PATH = "best.pt"
H5_CLASSES = ['Cheetah', 'Singa'] # Asumsi urutan kelas dalam model H5 Anda

# Cek apakah pustaka ML dapat dimuat. Jika gagal, inferensi nyata tidak akan berjalan.
try:
    # Mengimpor pustaka yang diperlukan untuk model nyata
    import tensorflow as tf
    from ultralytics import YOLO
    CAN_RUN_INFERENCE = True
except (ImportError, ModuleNotFoundError):
    CAN_RUN_INFERENCE = False

st.set_page_config(page_title="Visualisasi Model AI (Python Native)", page_icon="🦁", layout="wide")

# Styling CSS disingkat untuk tampilan
st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        .stButton>button { background-image: linear-gradient(to right, #43e97b 0%, #38f9d7 100%); color: #0d1117 !important; font-weight: bold; padding: 10px 20px; border-radius: 25px; border: none; box-shadow: 0 5px 20px rgba(67, 233, 123, 0.4); transition: all 0.3s ease; }
        .stButton>button:hover { box_shadow: 0 8px 30px rgba(56, 249, 215, 0.7); transform: translateY(-2px); }
        h1 { color: #ffffff; text-align: center; font-size: 2.5rem; text-shadow: 0 0 10px rgba(56, 249, 215, 0.5); }
        .subheader { color: #818cf8; text-align: center; margin-bottom: 2rem; font-style: italic; }
        .classification-card { background: linear-gradient(135deg, #7c3aed, #ec4899); color: white; padding: 1.5rem; border-radius: 15px; font-size: 1.6rem; font-weight: 700; text-align: center; box-shadow: 0 10px 20px rgba(124, 58, 237, 0.4); animation: fadeIn 1s; }
        .detection-item { background-color: #1f2937; padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 5px solid; display: flex; justify-content: space-between; align-items: center; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        /* Gaya untuk tampilan awal yang sudah dihapus, menyisakan style dasar */
        .welcome-box {
            background-color: #1a1e26; 
            padding: 30px;
            border-radius: 12px;
            border: 2px solid #818cf8;
            text-align: center;
            margin-top: 40px;
        }
        .welcome-title {
            color: #38f9d7;
            font-size: 1.8rem;
            margin-bottom: 10px;
        }
        .welcome-text {
            color: #c9d1d9;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- 2. PEMUATAN MODEL DENGAN CACHING ---
@st.cache_resource
def load_models(h5_path, pt_path):
    """Memuat model nyata. Jika gagal, akan mengembalikan None."""
    if CAN_RUN_INFERENCE:
        try:
            # Pemuatan Model Klasifikasi (TensorFlow/Keras)
            classifier = tf.keras.models.load_model(h5_path)
            # Pemuatan Model Deteksi (Ultralytics YOLO)
            detector = YOLO(pt_path)
            st.toast("Model ML nyata dimuat!", icon="✅")
            return classifier, detector
        except Exception as e:
            st.error(f"FATAL: Gagal memuat model nyata. Pastikan file model dan pustaka ML sudah tersedia. Error: {e}")
            return None, None
    else:
        st.error("FATAL: Pustaka TensorFlow atau Ultralytics tidak ditemukan. Inferensi model tidak dapat dijalankan.")
        return None, None

# Hanya muat model jika inferensi nyata dimungkinkan (jika tidak, ini hanya mengembalikan None)
CLASSIFIER, DETECTOR = load_models(H5_FILE_PATH, PT_FILE_PATH) 

# --- TAMPILAN JUDUL UTAMA ---
st.title("Aplikasi Klasifikasi & Deteksi Objek")

# Status Model sekarang menunjukkan apakah inferensi akan berjalan atau gagal (tidak ada lagi simulasi)
model_status_h5 = "Siap (Loaded)" if CLASSIFIER else f"Gagal ({H5_FILE_PATH})"
model_status_pt = "Siap (Loaded)" if DETECTOR else f"Gagal ({PT_FILE_PATH})"

st.markdown(f"""
    <p class="subheader">
        Penganalisis Hewan Buas Canggih: Model Deteksi 
        <span style="color:#38f9d7; font-weight:bold;">YOLOv8 ({model_status_pt})</span> & 
        Model Klasifikasi 
        <span style="color:#ec4899; font-weight:bold;">({model_status_h5})</span>
    </p>
""", unsafe_allow_html=True)


# --- 3. LOGIKA DETEKSI DAN GAMBAR BOUNDING BOX ---
def draw_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Menggambar bounding boxes pada gambar."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, int(height / 30))) 
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x_min, y_min, x_max, y_max = det["box"]
        box_color, label = det["color"], f"{det['class']} ({det['confidence']:.2f})"

        # Gambar Bounding Box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=box_color, width=8)

        # Gambar Label dengan latar belakang
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:4]
        y_start = max(0, y_min - text_h - 10)
        
        draw.rectangle([(x_min, y_start), (x_min + text_w + 15, y_min)], fill=box_color)
        draw.text((x_min + 7, y_start + 5), label, fill="#0d1117", font=font)
    return image


def process_image_with_models(uploaded_file, classifier, detector):
    """Fungsi untuk menjalankan inferensi nyata."""
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    
    # KARENA LOGIKA SIMULASI DIHAPUS, KITA HANYA MENGANDALKAN CAN_RUN_INFERENCE
    if classifier and detector and CAN_RUN_INFERENCE:
        # =========================================================
        # --- MODE INFERENSI NYATA ---
        # =========================================================
        
        # 1. Klasifikasi (TensorFlow/Keras)
        target_size = (224, 224) # Sesuaikan dengan ukuran input model H5 Anda
        img_resized = image.resize(target_size)
        img_array = np.array(img_resized) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch

        predictions = classifier.predict(img_array)
        class_index = np.argmax(predictions[0])
        sim_class = H5_CLASSES[class_index]
        sim_confidence = predictions[0][class_index]
        
        # 2. Deteksi Objek (YOLOv8)
        results = detector(image, verbose=False) 
        
        detections = []
        for r in results:
            if r.boxes.xyxy.numel() > 0:
                # Ambil hasil deteksi pertama saja
                box = r.boxes.xyxy[0].tolist()
                conf = r.boxes.conf[0].item()
                cls_idx = int(r.boxes.cls[0].item())
                
                det_class = r.names[cls_idx]
                
                color = '#38f9d7' if 'Cheetah' in det_class else '#ec4899'
                
                detections.append({
                    "class": det_class, 
                    "color": color, 
                    "box": [int(b) for b in box], 
                    "confidence": conf
                })
        
        # Jika YOLO tidak mendeteksi apa-apa, gunakan kotak sederhana sebagai fallback visual
        if not detections:
            detections = [{
                "class": sim_class, # Gunakan hasil klasifikasi sebagai fallback
                "color": '#818cf8', 
                "box": [int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)], 
                "confidence": sim_confidence
            }]
    
        # Hasil Klasifikasi
        classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"
            
        processed_image = draw_bounding_boxes(image.copy(), detections)
        
        return classification_result, detections, processed_image

    else:
        # PENTING: Jika inferensi NYATA gagal, kita mengembalikan None.
        # Ini akan memicu pesan peringatan di UI.
        return None, None, None


# --- 4. TAMPILAN STREAMLIT (ANTARMUKA PENGGUNA) ---

# Inisialisasi state jika belum ada
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
    
# Area untuk unggah file
uploaded_file = st.file_uploader(
    "Unggah Gambar Singa atau Cheetah Anda", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    help="Hanya jenis file gambar yang didukung."
)

# Tampilan Konten Utama
# Tampilan awal (Welcome State) telah dihapus

if uploaded_file:
    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ Pratinjau Gambar")
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

    with col_info:
        st.subheader("📊 Hasil Model")
        st.markdown('<p style="color:#ec4899; font-weight:600; font-size: 1.1rem;">[H5] Klasifikasi Utama</p>', unsafe_allow_html=True)
        placeholder_classification = st.empty()
        placeholder_classification.markdown('<div class="classification-card">...</div>', unsafe_allow_html=True)

        st.markdown('<br><p style="color:#38f9d7; font-weight:600; font-size: 1.1rem;">[PT] Deteksi Objek</p>', unsafe_allow_html=True)
        placeholder_detections = st.empty()
        placeholder_detections.info("Tekan tombol di bawah untuk menganalisis gambar.")

    st.markdown("---")
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        button_text = "▶️ Luncurkan Pemrosesan Model (NYATA)"
        
        if st.button(button_text, use_container_width=True):
            if CLASSIFIER and DETECTOR:
                with st.spinner("Memproses dengan model nyata..."):
                    st.session_state.classification, st.session_state.detections, st.session_state.processed_image = process_image_with_models(uploaded_file, CLASSIFIER, DETECTOR)
                st.toast("🎉 Pemrosesan Selesai! Visualisasi hasil model telah ditampilkan.")
            else:
                st.warning("Gagal memproses. Model tidak dimuat karena pustaka ML yang diperlukan tidak ditemukan. Anda harus menjalankan kode ini di lingkungan dengan TensorFlow dan Ultralytics.")


# --- 5. TAMPILKAN HASIL AKHIR ---
if 'processed_image' in st.session_state and st.session_state.processed_image:
    
    col_img, col_info = st.columns([2, 1])
    class_name = st.session_state.classification.split('(')[0].strip()
    emoji = "🦁" if "Singa" in class_name else "🐆" 

    with col_img:
        st.subheader("👁️ Visualisasi Deteksi")
        st.image(st.session_state.processed_image, caption=f"Hasil Deteksi: {class_name}", use_column_width=True)

    with col_info:
        st.subheader("📊 Hasil Model")
        
        # 1. Update Hasil Klasifikasi
        st.markdown('<p style="color:#ec4899; font-weight:600; font-size: 1.1rem;">[H5] Klasifikasi Utama</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="classification-card">{emoji} {st.session_state.classification}</div>', 
            unsafe_allow_html=True
        )

        # 2. Update Hasil Deteksi
        st.markdown('<br><p style="color:#38f9d7; font-weight:600; font-size: 1.1rem;">[PT] Deteksi Objek</p>', unsafe_allow_html=True)
        
        detection_list = ""
        for det in st.session_state.detections:
            color_code = det.get('color', '#818cf8') # Ambil warna dari deteksi, default jika tidak ada
            detection_list += f"""
                <div class="detection-item" style="border-left-color: {color_code};">
                    <span style="font-weight: bold; color: white;">{det['class']}</span>
                    <span style="background-color: {color_code}; color: #0d1117; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; font-weight: 700;">
                        Akurasi: {(det['confidence'] * 100):.1f}%
                    </span>
                </div>
            """
        st.markdown(f'<div style="animation: fadeIn 1.5s;">{detection_list}</div>', unsafe_allow_html=True)
