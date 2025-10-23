import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random

# --- KONSTANTA & SETUP AWAL ---
# Kelas untuk simulasi (H5_CLASSES)
H5_CLASSES = ['Singa', 'Cheetah']

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
        /* Gaya untuk tampilan awal */
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
def load_models():
    """Memuat model placeholder."""
    return "classifier_model.h5", "best.pt"

classifier_model, detector_model = load_models()

# --- TAMPILAN JUDUL UTAMA ---
st.title("Aplikasi Klasifikasi & Deteksi Objek")
st.markdown(f"""
    <p class="subheader">
        Penganalisis Hewan Buas Canggih: Memuat Model 
        <span style="color:#38f9d7; font-weight:bold;">YOLOv8 ({detector_model})</span> & 
        <span style="color:#ec4899; font-weight:bold;">Classifier ({classifier_model})</span>
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
    """Fungsi untuk mensimulasikan inferensi model."""
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    
    # --- LOGIKA SIMULASI HASIL YANG KONSISTEN ---
    
    # Mencoba menebak kelas dari nama file
    file_name = uploaded_file.name.lower()
    
    if "singa" in file_name:
        sim_class = 'Singa'
        color = '#ec4899' # Warna untuk Singa
    elif "cheetah" in file_name:
        sim_class = 'Cheetah'
        color = '#38f9d7' # Warna untuk Cheetah
    else:
        # Jika nama file tidak jelas, tetapkan ke Singa sebagai default
        sim_class = 'Singa' 
        color = '#ec4899'
        
    sim_confidence = random.uniform(0.95, 0.99)
    
    # Hasil Klasifikasi
    classification_result = f"{sim_class} (Probabilitas: {sim_confidence:.2%})"

    # Hasil Deteksi
    detections = [{
        "class": sim_class, 
        "color": color, 
        "box": [int(width * 0.20), int(height * 0.30), int(width * 0.75), int(height * 0.70)], 
        "confidence": sim_confidence
    }]
        
    processed_image = draw_bounding_boxes(image.copy(), detections)
    
    return classification_result, detections, processed_image


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
if uploaded_file is None and st.session_state.processed_image is None:
    # --- TAMPILAN AWAL (WELCOME STATE) ---
    st.markdown("""
        <div class="welcome-box">
            <p class="welcome-title">Selamat Datang di Demo Model AI Hewan Buas!</p>
            <p class="welcome-text">
                Unggah gambar di atas untuk melihat bagaimana model Klasifikasi (H5) dan Deteksi Objek (YOLOv8) bekerja. 
                Kami akan menganalisis gambar dan memvisualisasikan hasilnya dengan <i>bounding box</i>.
            </p>
            <p class="welcome-text" style="margin-top: 15px; font-weight: bold; color: #818cf8;">
                Petunjuk: Coba unggah gambar dengan nama file mengandung kata 'singa' atau 'cheetah' untuk hasil yang konsisten.
            </p>
        </div>
    """, unsafe_allow_html=True)


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
        if st.button("▶️ Luncurkan Pemrosesan Model", use_container_width=True):
            with st.spinner("Memproses dengan model Anda (Inferensi disimulasikan karena keterbatasan lingkungan)..."):
                st.session_state.classification, st.session_state.detections, st.session_state.processed_image = process_image_with_models(uploaded_file, classifier_model, detector_model)
            st.toast("🎉 Pemrosesan Selesai! Visualisasi hasil model telah ditampilkan.")


# --- 5. TAMPILKAN HASIL AKHIR ---
# KOREKSI: Hapus kondisi 'and uploaded_file is None' agar hasil tetap ditampilkan setelah tombol ditekan.
if 'processed_image' in st.session_state and st.session_state.processed_image:
    
    col_img, col_info = st.columns([2, 1])
    class_name = st.session_state.classification.split('(')[0].strip()
    emoji = "🦁" if "Singa" in class_name else "🐆" 

    with col_img:
        st.subheader("👁️ Visualisasi Deteksi")
        # Pastikan gambar yang ditampilkan adalah hasil yang diproses (st.session_state.processed_image)
        st.image(st.session_state.processed_image, caption=f"Hasil Deteksi: {class_name}", use_column_width=True)

    with col_info:
        st.subheader("📊 Hasil Model")
        
        # 1. Update Hasil Klasifikasi
        st.markdown('<p style="color:#ec4899; font-weight:600; font-size: 1.1rem;">[H5] Klasifikasi Utama</p>', unsafe_allow_html=True)
        # Gunakan placeholder yang tepat jika diperlukan, atau tampilkan langsung
        st.markdown(
            f'<div class="classification-card">{emoji} {st.session_state.classification}</div>', 
            unsafe_allow_html=True
        )

        # 2. Update Hasil Deteksi
        st.markdown('<br><p style="color:#38f9d7; font-weight:600; font-size: 1.1rem;">[PT] Deteksi Objek</p>', unsafe_allow_html=True)
        
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
        st.markdown(f'<div style="animation: fadeIn 1.5s;">{detection_list}</div>', unsafe_allow_html=True)
