import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import random
import numpy as np

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
        Simulasi visualisasi hasil dari model 
        <span style="color:#a3e635; font-weight:bold;">YOLOv8 (best.pt)</span> dan 
        <span style="color:#f472b6; font-weight:bold;">Classifier (classifier_model.h5)</span>.
        <br>
        Output simulasi terbatas hanya pada deteksi **Singa** atau **Cheetah**.
    </p>
""", unsafe_allow_html=True)


# --- 2. Data Simulasi (Mock Data) ---

# Data simulasi yang hanya memiliki Singa ATAU Cheetah
MOCK_SCENARIOS = [
    {
        "name": "Singa Tunggal",
        "classification": "Singa (Probabilitas: 99.8%)",
        "detections": [
            # Bounding box dalam koordinat relatif (0-1)
            # Format: [x_min, y_min, x_max, y_max]
            {"class": 'Singa', "color": 'orangered', "box": [0.20, 0.30, 0.75, 0.70], "confidence": 0.98}, 
        ]
    },
    {
        "name": "Cheetah Berburu",
        "classification": "Cheetah (Probabilitas: 97.5%)",
        "detections": [
            # Bounding box dalam koordinat relatif (0-1)
            {"class": 'Cheetah', "color": 'gold', "box": [0.15, 0.45, 0.80, 0.85], "confidence": 0.95},
        ]
    }
]


# --- 3. Fungsi Logika (Deteksi dan Gambar Bounding Box) ---

def draw_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Menggambar bounding boxes dan label pada gambar menggunakan PIL."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Coba memuat font yang lebih baik jika tersedia, jika tidak gunakan default
    try:
        font = ImageFont.truetype("arial.ttf", size=24) 
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        # Konversi koordinat relatif (0-1) ke koordinat piksel
        x_min = int(det["box"][0] * width)
        y_min = int(det["box"][1] * height)
        x_max = int(det["box"][2] * width)
        y_max = int(det["box"][3] * height)
        
        box_color = det["color"]
        label = f"{det['class']} ({det['confidence']:.2f})"

        # 1. Gambar Bounding Box
        # Streamlit dan PIL adalah cara Python untuk melakukan apa yang dilakukan Canvas di HTML
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=box_color, 
            width=5
        )

        # 2. Gambar Label dengan latar belakang
        text_bbox = draw.textbbox((x_min, y_min), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Gambar latar belakang label
        draw.rectangle(
            [(x_min, y_min - text_height - 5), (x_min + text_width + 10, y_min)], 
            fill=box_color
        )
        
        # Gambar teks label
        draw.text(
            (x_min + 5, y_min - text_height - 3), 
            label, 
            fill="black", # Teks hitam untuk kontras yang lebih baik
            font=font
        )
    return image


def process_image_simulated(uploaded_file):
    """Fungsi utama untuk memproses gambar dengan simulasi model."""
    
    # 1. Memuat Gambar
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. Simulasi Model: Pilih hasil acak (Singa atau Cheetah)
    mock_results = random.choice(MOCK_SCENARIOS)
    
    # 3. Mendapatkan Hasil
    classification_result = mock_results["classification"]
    detections = mock_results["detections"]
    
    # 4. Menggambar Bounding Box
    processed_image = draw_bounding_boxes(image, detections)
    
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
        placeholder_detections.info("Tekan tombol di bawah untuk menjalankan simulasi model.")

    st.markdown("---")
    
    # Tombol Proses di area terpisah
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("▶️ Luncurkan Pemrosesan Model (Simulasi)", use_container_width=True):
            
            # Tampilkan loading state
            with st.spinner("Memproses (Simulasi Model 3 detik)..."):
                # Panggil fungsi simulasi
                st.session_state.classification, st.session_state.detections, st.session_state.processed_image = process_image_simulated(uploaded_file)
            
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
    # Inisialisasi state jika belum ada
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None


# --- 6. Footer ---

st.sidebar.markdown("# Petunjuk")
st.sidebar.info("Aplikasi ini mensimulasikan alur kerja klasifikasi dan deteksi objek. Saat Anda menekan tombol proses, kode Python akan: \n\n1. Memilih hasil model secara acak (Singa atau Cheetah).\n2. Menggunakan pustaka **PIL (Pillow)** untuk menggambar kotak pembatas pada gambar.\n3. Menampilkan hasilnya.")
