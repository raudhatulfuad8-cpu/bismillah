import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import random
import numpy as np

# --- WARNING: PENANGANAN DEPENDENSI MODEL ---
# Catatan: Untuk menjalankan model .h5 dan .pt yang sebenarnya, 
# Anda perlu memastikan pustaka 'tensorflow', 'ultralytics', dan 'opencv-python' 
# terinstal di lingkungan Streamlit Anda. Karena keterbatasan lingkungan ini, 
# kode di bawah hanya memuat struktur model dan mensimulasikan inferensi.
try:
    # Coba impor pustaka yang diperlukan (untuk penulisan kode yang benar)
    import tensorflow as tf
    # from tensorflow.keras.models import load_model
    # from ultralytics import YOLO
    # import cv2
    HAS_MODEL_LIBS = True
except ImportError:
    HAS_MODEL_LIBS = False
    
# Kelas yang dilatih untuk model klasifikasi H5
# Diasumsikan model H5 melatih dua kategori ini.
H5_CLASSES = ['Singa', 'Cheetah']
# --- AKHIR PENANGANAN DEPENDENSI ---


# --- 1. Konfigurasi Halaman dan Styling (Python & Streamlit) ---

st.set_page_config(
    page_title="Visualisasi Model AI (Python Native)",
    page_icon="🦁",
    layout="wide"
)

# Styling yang DITINGKATKAN untuk visualisasi yang lebih menarik
st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117; /* Darker background */
            color: #c9d1d9;
        }
        /* Peningkatan Styling untuk Tombol */
        .stButton>button {
            background-image: linear-gradient(to right, #43e97b 0%, #38f9d7 100%); /* Neon Green/Cyan Gradient */
            color: #0d1117 !important;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 25px; /* Lebih Bulat */
            border: none;
            box-shadow: 0 5px 20px rgba(67, 233, 123, 0.4); /* Glow effect */
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box_shadow: 0 8px 30px rgba(56, 249, 215, 0.7);
            transform: translateY(-2px);
        }
        /* Peningkatan Gaya Judul */
        h1 {
            color: #ffffff;
            text-align: center;
            font-size: 2.5rem;
            text-shadow: 0 0 10px rgba(56, 249, 215, 0.5); /* Efek Cahaya */
        }
        .subheader {
            color: #818cf8; /* Biru/Ungu lembut */
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }
        /* Kotak Hasil Klasifikasi yang Lebih Menarik (Card) */
        .classification-card {
            background: linear-gradient(135deg, #7c3aed, #ec4899); /* Gradien Ungu-Pink */
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            font-size: 1.6rem;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 10px 20px rgba(124, 58, 237, 0.4);
            animation: fadeIn 1s;
        }
        /* Kotak Hasil Deteksi (List Item) */
        .detection-item {
            background-color: #1f2937;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            border-left: 5px solid; /* Solid line for color coding */
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        /* Animasi Sederhana */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)


# --- 2. Pemuatan Model dengan Caching ---

@st.cache_resource
def load_models():
    """Memuat model H5 dan PT hanya sekali menggunakan Streamlit caching."""
    # Menghapus seluruh pesan st.success/st.error tentang Pustaka ML dari sidebar/main page
    # if HAS_MODEL_LIBS:
    #     st.success("✅ Pustaka ML (TensorFlow/Ultralytics) terdeteksi. Siap untuk inferensi nyata!")
    # else:
    #     st.error("Peringatan: Pustaka TensorFlow/Ultralytics tidak ditemukan. Inferensi model akan disimulasikan.")

    # Catatan: Kami hanya menggunakan path file sebagai placeholder.
    H5_FILE_PATH = "classifier_model.h5"
    PT_FILE_PATH = "best.pt"

    # Placeholder model object
    classifier = H5_FILE_PATH 
    detector = PT_FILE_PATH

    # --- Bagian status pemuatan model di sidebar telah dihapus di sini ---
    # st.sidebar.markdown("---")
    # st.sidebar.success(f"Model Klasifikasi ({H5_FILE_PATH}) dimuat (placeholder).")
    # st.sidebar.success(f"Model Deteksi ({PT_FILE_PATH}) dimuat (placeholder).")
        
    return classifier, detector

# Muat model di awal aplikasi
classifier_model, detector_model = load_models()


# --- Tampilkan Judul Utama setelah Model dimuat ---
# Perbaikan: Variabel model sekarang sudah didefinisikan di sini.
st.title("Aplikasi Klasifikasi & Deteksi Objek")
st.markdown(f"""
    <p class="subheader">
        Penganalisis Hewan Buas Canggih: Memuat Model 
        <span style="color:#38f9d7; font-weight:bold;">YOLOv8 ({detector_model})</span> & 
        <span style="color:#ec4899; font-weight:bold;">Classifier ({classifier_model})</span>
    </p>
""", unsafe_allow_html=True)


# --- 3. Fungsi Logika (Deteksi dan Gambar Bounding Box) ---

def draw_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Menggambar bounding boxes dan label pada gambar menggunakan PIL."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Coba memuat font yang lebih baik jika tersedia, jika tidak gunakan default
    try:
        font_size = max(16, int(height / 30))
        font = ImageFont.truetype("arial.ttf", size=font_size) 
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x_min = det["box"][0]
        y_min = det["box"][1]
        x_max = det["box"][2]
        y_max = det["box"][3]
        
        box_color = det["color"]
        label = f"{det['class']} ({det['confidence']:.2f})"

        # 1. Gambar Bounding Box (Diisi dengan warna transparan, outline tebal)
        # Membuat kotak semi-transparan (perlu menggunakan ImageDraw.Draw pada gambar RGB atau RGBA)
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=box_color, 
            width=8 # Tebal
        )

        # 2. Gambar Label dengan latar belakang
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        y_start = max(0, y_min - text_height - 10)
        
        # Gambar latar belakang label
        draw.rectangle(
            [(x_min, y_start), (x_min + text_width + 15, y_min)], 
            fill=box_color
        )
        
        # Gambar teks label
        draw.text(
            (x_min + 7, y_start + 5), 
            label, 
            fill="#0d1117", # Teks gelap pada latar terang
            font=font
        )
    return image


def process_image_with_models(uploaded_file, classifier, detector):
    """
    Fungsi untuk menjalankan inferensi dengan model H5 dan PT yang sebenarnya.
    (Saat ini mensimulasikan hasil dan mengembalikan format data yang diharapkan.)
    """
    # 1. Memuat Gambar dari Streamlit Uploader
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    
    sim_class = random.choice(H5_CLASSES)
    sim_confidence = random.uniform(0.95, 0.99)

    # --- INFERENSI KLASIFIKASI (Model H5) ---
    if classifier and HAS_MODEL_LIBS:
        # KODE INFERENSI NYATA H5 DI SINI
        # ...
        classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"
    else: 
        # Simulasi hasil H5:
        classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"


    # --- INFERENSI DETEKSI OBJEK (Model PT/YOLO) ---
    detections = []
    
    if detector and HAS_MODEL_LIBS:
        # KODE INFERENSI NYATA YOLO DI SINI
        # ...
        # JIKA BERHASIL DIMUAT, TAPI INFERENSI TETAP DISIMULASIKAN
        color = '#38f9d7' if sim_class == 'Cheetah' else '#ec4899'
        detections_simulated = [{
            "class": sim_class, 
            "color": color, 
            "box": [
                int(width * 0.20), int(height * 0.30), 
                int(width * 0.75), int(height * 0.70)
            ], 
            "confidence": sim_confidence
        }]
        detections = detections_simulated
    
    else: # Model dimuat (placeholder), tapi libs tidak ada.
        # Simulasi hasil Deteksi:
        color = '#38f9d7' if sim_class == 'Cheetah' else '#ec4899'
        detections_simulated = [{
            "class": sim_class, 
            "color": color, 
            "box": [
                int(width * 0.20), int(height * 0.30), 
                int(width * 0.75), int(height * 0.70)
            ], 
            "confidence": sim_confidence
        }]
        detections = detections_simulated
        
    # 4. Menggambar Bounding Box
    processed_image = draw_bounding_boxes(image.copy(), detections)
    
    return classification_result, detections, processed_image


# --- 4. Tampilan Streamlit (Antarmuka Pengguna) ---

# Area untuk unggah file
uploaded_file = st.file_uploader(
    "Unggah Gambar Singa atau Cheetah Anda", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    help="Hanya jenis file gambar yang didukung."
)

if uploaded_file is not None:
    
    # 1. Pratinjau Gambar Asli
    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ Pratinjau Gambar")
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

    with col_info:
        st.subheader("📊 Hasil Model")
        
        # Kontainer Hasil Klasifikasi (Model H5)
        st.markdown('<p style="color:#ec4899; font-weight:600; font-size: 1.1rem;">[H5] Klasifikasi Utama</p>', unsafe_allow_html=True)
        placeholder_classification = st.empty()
        placeholder_classification.markdown('<div class="classification-card">...</div>', unsafe_allow_html=True)

        # Kontainer Hasil Deteksi (YOLOv8)
        st.markdown('<br><p style="color:#38f9d7; font-weight:600; font-size: 1.1rem;">[PT] Deteksi Objek</p>', unsafe_allow_html=True)
        placeholder_detections = st.empty()
        placeholder_detections.info("Tekan tombol di bawah untuk menganalisis gambar.")

    st.markdown("---")
    
    # Tombol Proses di area terpisah
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("▶️ Luncurkan Pemrosesan Model", use_container_width=True):
            
            # Tampilkan loading state
            with st.spinner("Memproses dengan model Anda (Inferensi disimulasikan karena keterbatasan lingkungan)..."):
                # Panggil fungsi pemrosesan model
                st.session_state.classification, st.session_state.detections, st.session_state.processed_image = process_image_with_models(uploaded_file, classifier_model, detector_model)
            
            st.toast("🎉 Pemrosesan Selesai! Visualisasi hasil model telah ditampilkan.")


# --- 5. Tampilkan Hasil Akhir (Setelah Pemrosesan) ---

if 'processed_image' in st.session_state and st.session_state.processed_image is not None:
    # Mengganti pratinjau dengan gambar yang sudah diproses
    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("👁️ Visualisasi Deteksi")
        # Menampilkan gambar yang sudah digambar bounding box-nya
        st.image(st.session_state.processed_image, caption=f"Hasil Deteksi: {st.session_state.classification.split(' ')[0]}", use_column_width=True)

    with col_info:
        st.subheader("📊 Hasil Model")
        
        # 1. Update Hasil Klasifikasi
        st.markdown('<p style="color:#ec4899; font-weight:600; font-size: 1.1rem;">[H5] Klasifikasi Utama</p>', unsafe_allow_html=True)
        
        class_name = st.session_state.classification.split('(')[0].strip()
        emoji = "🦁" if "Singa" in class_name else "🐆" # Gunakan emoji untuk visual
        
        placeholder_classification.markdown(
            f'<div class="classification-card">{emoji} {st.session_state.classification}</div>', 
            unsafe_allow_html=True
        )

        # 2. Update Hasil Deteksi
        st.markdown('<br><p style="color:#38f9d7; font-weight:600; font-size: 1.1rem;">[PT] Deteksi Objek</p>', unsafe_allow_html=True)
        
        if st.session_state.detections:
            detection_list = ""
            for det in st.session_state.detections:
                color_code = '#38f9d7' if "Cheetah" in det['class'] else '#ec4899'
                detection_list += f"""
                    <div class="detection-item" style="border-left-color: {color_code};">
                        <span style="font-weight: bold; color: white;">{det['class']}</span>
                        <span style="background-color: {color_code}; color: #0d1117; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; font-weight: 700;">
                            Akurasi: {(det['confidence'] * 100):.1f}%
                        </span>
                    </div>
                """
            placeholder_detections.markdown(f'<div style="animation: fadeIn 1.5s;">{detection_list}</div>', unsafe_allow_html=True)
        else:
            placeholder_detections.warning("Tidak ada objek yang terdeteksi atau model deteksi gagal dimuat.")

else:
    # Inisialisasi state jika belum ada
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
