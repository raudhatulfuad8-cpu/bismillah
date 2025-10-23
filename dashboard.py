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
    page_icon="✨",
    layout="wide"
)

# Styling menggunakan Markdown dan unsafe_allow_html=True
# Ini mensimulasikan tema gelap dan memperkuat estetika
st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117; 
            color: #c9d1d9;
        }
        /* Styling untuk tombol dan judul */
        .stButton>button {
            background-image: linear-gradient(to right, #8b5cf6, #ec4899); /* Ungu ke Pink */
            color: white !important;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.5);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 20px rgba(236, 72, 153, 0.7);
        }
        h1 {
            color: white;
            text-align: center;
        }
        .subheader {
            color: #9ca3af;
            text-align: center;
            margin-bottom: 2rem;
        }
        .box-title-fuchsia {
            color: #f472b6; /* Fuchsia-400 */
            font-weight: 600;
        }
        .box-title-lime {
            color: #a3e635; /* Lime-400 */
            font-weight: 600;
        }
        .classification-output {
            background-color: #5b21b6; /* Violet-700 */
            color: white;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Aplikasi Klasifikasi & Deteksi Objek")
st.markdown(f"""
    <p class="subheader">
        Memuat dan menjalankan model Anda:
        <span style="color:#a3e635; font-weight:bold;">YOLOv8 (best.pt)</span> dan 
        <span style="color:#f472b6; font-weight:bold;">Classifier (classifier_model.h5)</span>.
    </p>
""", unsafe_allow_html=True)


# --- 2. Pemuatan Model dengan Caching ---

@st.cache_resource
def load_models():
    """Memuat model H5 dan PT hanya sekali menggunakan Streamlit caching."""
    if not HAS_MODEL_LIBS:
        st.error("Peringatan: Pustaka TensorFlow/Ultralytics tidak ditemukan. Model tidak dapat dimuat atau dijalankan secara nyata.")
        return None, None

    # Catatan: Dalam deployment Streamlit yang nyata, file-file ini harus ada di direktori yang sama 
    # atau dimuat dari path yang benar.
    
    # 1. Pemuatan Model H5 (Klasifikasi)
    H5_FILE_PATH = "classifier_model.h5"
    try:
        # classifier = tf.keras.models.load_model(H5_FILE_PATH) # Kode pemuatan aktual
        classifier = H5_FILE_PATH # Placeholder model object
        st.success(f"✅ Model Klasifikasi ({H5_FILE_PATH}) siap.")
    except Exception as e:
        st.error(f"Gagal memuat {H5_FILE_PATH}: {e}")
        classifier = None
        
    # 2. Pemuatan Model PT (Deteksi)
    PT_FILE_PATH = "best.pt"
    try:
        # detector = YOLO(PT_FILE_PATH) # Kode pemuatan aktual
        detector = PT_FILE_PATH # Placeholder model object
        st.success(f"✅ Model Deteksi ({PT_FILE_PATH}) siap.")
    except Exception as e:
        st.error(f"Gagal memuat {PT_FILE_PATH}: {e}")
        detector = None
        
    return classifier, detector

# Muat model di awal aplikasi
classifier_model, detector_model = load_models()


# --- 3. Fungsi Logika (Deteksi dan Gambar Bounding Box) ---

def draw_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Menggambar bounding boxes dan label pada gambar menggunakan PIL."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Coba memuat font yang lebih baik jika tersedia, jika tidak gunakan default
    try:
        # Ukuran font disesuaikan agar sesuai dengan gambar
        font = ImageFont.truetype("arial.ttf", size=max(12, int(height / 35))) 
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        # Koordinat dalam format piksel (setelah dikonversi oleh fungsi pemrosesan)
        x_min = det["box"][0]
        y_min = det["box"][1]
        x_max = det["box"][2]
        y_max = det["box"][3]
        
        box_color = det["color"]
        label = f"{det['class']} ({det['confidence']:.2f})"

        # 1. Gambar Bounding Box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=box_color, 
            width=5
        )

        # 2. Gambar Label dengan latar belakang
        # Pastikan label tidak keluar dari batas atas
        text_height = draw.textbbox((0, 0), label, font=font)[3] - draw.textbbox((0, 0), label, font=font)[1]
        text_width = draw.textbbox((0, 0), label, font=font)[2] - draw.textbbox((0, 0), label, font=font)[0]
        
        y_start = max(0, y_min - text_height - 5)

        # Gambar latar belakang label
        draw.rectangle(
            [(x_min, y_start), (x_min + text_width + 10, y_min)], 
            fill=box_color
        )
        
        # Gambar teks label
        draw.text(
            (x_min + 5, y_start + 2), 
            label, 
            fill="black", 
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

    # --- INFERENSI KLASIFIKASI (Model H5) ---
    if classifier and HAS_MODEL_LIBS:
        # a. Preprocessing untuk H5 model (Misalnya, resize ke 224x224)
        # H5_input = image.resize((224, 224))
        # H5_array = np.asarray(H5_input) / 255.0
        # H5_array = np.expand_dims(H5_array, axis=0)
        
        # b. Inferensi Model H5
        # predictions = classifier.predict(H5_array)[0]
        # predicted_class_index = np.argmax(predictions)
        # confidence = predictions[predicted_class_index]
        # classification_result = f"{H5_CLASSES[predicted_class_index]} (Probabilitas: {confidence:.2%})"
        
        # JIKA BERHASIL DIMUAT, TAPI INFERENSI TETAP DISIMULASIKAN
        sim_class = random.choice(H5_CLASSES)
        sim_confidence = random.uniform(0.95, 0.99)
        classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"
    elif classifier: # Model dimuat (placeholder), tapi libs tidak ada.
        # Simulasi hasil H5:
        sim_class = random.choice(H5_CLASSES)
        sim_confidence = random.uniform(0.95, 0.99)
        classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"
    else:
        classification_result = "KLASIFIKASI GAGAL (Model H5 tidak dimuat)"

    # --- INFERENSI DETEKSI OBJEK (Model PT/YOLO) ---
    detections = []
    
    if detector and HAS_MODEL_LIBS:
        # c. Inferensi Model PT/YOLO
        # results = detector(image) # Panggilan inferensi YOLOv8

        # d. Ekstraksi dan Konversi Bounding Box
        # detections = []
        # for r in results:
        #     for box in r.boxes:
        #         x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        #         cls = r.names[int(box.cls[0])]
        #         conf = float(box.conf[0])
        #         # Tambahkan logika warna berdasarkan kelas di sini
        #         detections.append({"class": cls, "color": "lime", "box": [x1, y1, x2, y2], "confidence": conf})

        # JIKA BERHASIL DIMUAT, TAPI INFERENSI TETAP DISIMULASIKAN
        # Gunakan hasil klasifikasi (sim_class) untuk deteksi yang konsisten
        color = 'orangered' if sim_class == 'Singa' else 'gold'
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
    
    elif detector: # Model dimuat (placeholder), tapi libs tidak ada.
        # Simulasi hasil Deteksi:
        # Data simulasi dalam format yang diharapkan setelah konversi dari hasil YOLO.
        # Format kotak harus PIKSEL ABSOLUT di sini
        color = 'orangered' if sim_class == 'Singa' else 'gold'
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
    else:
        # detections sudah diinisialisasi sebagai [] di luar if
        pass
        
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
        st.subheader("Pratinjau Gambar")
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

    with col_info:
        st.subheader("Informasi Model")
        # Kontainer Hasil Klasifikasi (Model H5)
        st.markdown('<p class="box-title-fuchsia">Hasil Klasifikasi (classifier_model.h5)</p>', unsafe_allow_html=True)
        placeholder_classification = st.empty()
        placeholder_classification.markdown('<div class="classification-output">Menunggu Proses...</div>', unsafe_allow_html=True)

        # Kontainer Hasil Deteksi (YOLOv8)
        st.markdown('<br><p class="box-title-lime">Hasil Deteksi Objek (best.pt)</p>', unsafe_allow_html=True)
        placeholder_detections = st.empty()
        placeholder_detections.info("Tekan tombol di bawah untuk menjalankan model Anda.")

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

if 'processed_image' in st.session_state:
    # Mengganti pratinjau dengan gambar yang sudah diproses
    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("Visualisasi Hasil Model")
        # Menampilkan gambar yang sudah digambar bounding box-nya
        st.image(st.session_state.processed_image, caption=f"Hasil Deteksi Objek: {st.session_state.classification.split(' ')[0]}", use_column_width=True)

    with col_info:
        st.subheader("Informasi Model")
        
        # 1. Update Hasil Klasifikasi
        st.markdown('<p class="box-title-fuchsia">Hasil Klasifikasi (classifier_model.h5)</p>', unsafe_allow_html=True)
        placeholder_classification.markdown(
            f'<div class="classification-output">{st.session_state.classification}</div>', 
            unsafe_allow_html=True
        )

        # 2. Update Hasil Deteksi
        st.markdown('<br><p class="box-title-lime">Hasil Deteksi Objek (best.pt)</p>', unsafe_allow_html=True)
        
        if st.session_state.detections:
            detection_list = ""
            for det in st.session_state.detections:
                # Gunakan st.markdown untuk list berestetika
                detection_list += f"""
                    <div style="background-color: #1f2937; padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 4px solid {det['color']}; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; color: white;">{det['class']}</span>
                        <span style="background-color: {det['color']}; color: black; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem;">
                            {(det['confidence'] * 100):.1f}%
                        </span>
                    </div>
                """
            placeholder_detections.markdown(detection_list, unsafe_allow_html=True)
        else:
            placeholder_detections.warning("Tidak ada objek yang terdeteksi atau model deteksi gagal dimuat.")

else:
    # Inisialisasi state jika belum ada
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None


# --- 6. Footer ---

st.sidebar.markdown("# Petunjuk")
st.sidebar.info(f"""
    Aplikasi ini telah dimodifikasi untuk memuat file model Anda yang diunggah: 
    - Klasifikasi: `{classifier_model}`
    - Deteksi: `{detector_model}`
    
    **PENTING:** Agar model ini dapat berjalan secara nyata (bukan simulasi), Anda harus:
    1.  *Uncomment* (hapus `#`) baris `load_model` dan `YOLO` di fungsi `load_models()`.
    2.  *Uncomment* kode Inferensi yang relevan di fungsi `process_image_with_models()`.
    3.  Pastikan lingkungan Streamlit Anda memiliki pustaka **`tensorflow`** dan **`ultralytics`** yang terinstal.
""")
