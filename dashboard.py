import streamlit as st
import streamlit.components.v1 as components
import random

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Visualisasi Model AI",
    page_icon="✨",
    layout="wide"
)

# Menghilangkan padding atas dan samping default untuk tampilan penuh
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# --- KODE HTML/JS LENGKAP DARI VISUALISASI DI BAWAH INI (Telah Dimodifikasi) ---
html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisasi Deteksi Objek & Klasifikasi Gambar</title>
    <!-- Muat Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Konfigurasi Font Inter -->
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0d1117; /* Latar belakang gelap */
            color: #c9d1d9;
        }
        /* Efek glow pada batas card utama */
        #app {
            box-shadow: 0 0 25px rgba(255, 105, 180, 0.2); /* Bayangan pink/fuchsia lembut */
        }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8 flex items-center justify-center">

    <div id="app" class="w-full max-w-4xl bg-gray-900/90 p-6 md:p-10 rounded-xl border border-fuchsia-800/50 transition-all duration-500 hover:shadow-fuchsia-500/30">
        <h1 class="text-4xl font-extrabold mb-4 text-white text-center tracking-tight">
            Aplikasi Klasifikasi & Deteksi Objek
        </h1>
        <p class="text-center mb-8 text-gray-400">
            Visualisasi hasil dari model <span class="text-lime-400 font-semibold">YOLOv8 (.pt)</span> dan <span class="text-fuchsia-400 font-semibold">Classifier (.h5)</span>.
        </p>

        <!-- Area Unggah File -->
        <div class="mb-8 p-6 border-2 border-dashed border-pink-500/50 bg-gray-800 rounded-xl 
              hover:border-pink-400 hover:shadow-md hover:shadow-pink-500/20 transition duration-300">
            <input type="file" id="imageUpload" accept="image/*" class="hidden" onchange="previewImage(event)">
            <label for="imageUpload" class="cursor-pointer flex flex-col items-center justify-center py-6 text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-10 h-10 mb-2 text-pink-400 animate-pulse">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                </svg>
                <span class="text-white font-bold text-lg">Klik untuk Unggah Gambar</span>
                <span class="text-sm">atau seret dan lepas gambar (JPG, PNG)</span>
            </label>
        </div>

        <!-- Area Tampilan Gambar dan Hasil -->
        <div id="resultsArea" class="hidden">
            <div class="relative mb-8 rounded-xl overflow-hidden border border-gray-700 shadow-xl shadow-gray-700/20">
                <img id="uploadedImage" src="#" alt="Gambar Terunggah" class="w-full h-auto object-contain max-h-[500px]">
                <canvas id="detectionCanvas" class="absolute top-0 left-0 w-full h-full"></canvas>
            </div>

            <div class="flex flex-col md:flex-row gap-6">
                <!-- Klasifikasi Gambar (H5 Model) -->
                <div class="flex-1 p-5 bg-gray-800 rounded-xl border border-fuchsia-600/50 shadow-lg shadow-fuchsia-500/10">
                    <h2 class="text-xl font-semibold mb-3 text-fuchsia-400">1. Hasil Klasifikasi Gambar (.h5)</h2>
                    <p class="text-sm text-gray-500 mb-4">Klasifikasi Kategori Global (Classifier Model)</p>
                    <div id="classificationResult" class="text-3xl font-extrabold text-white bg-fuchsia-800/50 p-4 rounded-xl text-center border-2 border-fuchsia-500/80">
                        Memuat...
                    </div>
                </div>

                <!-- Deteksi Objek (YOLOv8 Model) -->
                <div class="flex-1 p-5 bg-gray-800 rounded-xl border border-lime-600/50 shadow-lg shadow-lime-500/10">
                    <h2 class="text-xl font-semibold mb-3 text-lime-400">2. Hasil Deteksi Objek (YOLOv8)</h2>
                    <p class="text-sm text-gray-500 mb-4">Objek yang Ditemukan dan Lokasinya (YOLOv8 Model)</p>
                    <ul id="detectionList" class="space-y-3 max-h-40 overflow-y-auto">
                        <!-- Hasil deteksi akan dimasukkan di sini -->
                        <li class="text-white text-lg text-center py-4">Memuat...</li>
                    </ul>
                </div>
            </div>

            <!-- Tombol Proses dengan Gradien -->
            <button id="processButton" onclick="processImage()" class="w-full mt-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-extrabold text-lg rounded-xl transition duration-300 shadow-xl shadow-pink-500/50 disabled:opacity-50 disabled:shadow-none" disabled>
                <span id="buttonText">Luncurkan Pemrosesan Model</span>
                <span id="loadingSpinner" class="hidden">
                    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Memproses Model...
                </span>
            </button>
        </div>

        <!-- Pesan Modal untuk Simulasi -->
        <div id="messageBox" class="fixed inset-0 bg-black bg-opacity-75 hidden items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-8 rounded-xl border border-lime-500 shadow-2xl max-w-sm text-center">
                <p id="messageContent" class="text-xl font-bold text-lime-400 mb-4"></p>
                <button onclick="document.getElementById('messageBox').classList.add('hidden')" class="bg-lime-600 hover:bg-lime-700 text-black font-semibold py-2 px-6 rounded-lg transition duration-300">Oke, Mengerti</button>
            </div>
        </div>

    </div>

    <script>
        // Variabel global
        let uploadedImageElement;
        let detectionCanvas;
        let processButton;
        let classificationResultElement;
        let detectionListElement;
        let resultsArea;
        let uploadedFile;
        let currentDetections = []; 

        // Data hasil tunggal (Lion atau Cheetah) yang akan diisi saat proses
        let currentMockResult = {}; 

        document.addEventListener('DOMContentLoaded', () => {
            uploadedImageElement = document.getElementById('uploadedImage');
            detectionCanvas = document.getElementById('detectionCanvas');
            processButton = document.getElementById('processButton');
            classificationResultElement = document.getElementById('classificationResult');
            detectionListElement = document.getElementById('detectionList');
            resultsArea = document.getElementById('resultsArea');
        });

        // Fungsi untuk menampilkan pesan modal (pengganti alert())
        function showMessage(content) {
            document.getElementById('messageContent').textContent = content;
            document.getElementById('messageBox').classList.remove('hidden');
            document.getElementById('messageBox').classList.add('flex');
        }

        // 1. Pratinjau Gambar dan Set up UI
        function previewImage(event) {
            uploadedFile = event.target.files[0];
            if (uploadedFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    // Tampilkan area hasil
                    resultsArea.classList.remove('hidden');

                    // Set sumber gambar
                    uploadedImageElement.src = e.target.result;

                    // Bersihkan canvas
                    const ctx = detectionCanvas.getContext('2d');
                    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                    
                    // Reset daftar deteksi saat ini
                    currentDetections = [];
                    currentMockResult = {};

                    // Bersihkan hasil sebelumnya
                    classificationResultElement.textContent = 'Menunggu Pemrosesan...';
                    classificationResultElement.classList.remove('bg-fuchsia-800/50', 'border-fuchsia-500/80');
                    classificationResultElement.classList.add('bg-gray-700/30', 'border-gray-600');
                    detectionListElement.innerHTML = '<li class="text-gray-400 text-md text-center py-4">Tekan tombol proses untuk menjalankan model.</li>';

                    // Aktifkan tombol proses
                    processButton.disabled = false;
                    document.getElementById('buttonText').classList.remove('hidden');
                    document.getElementById('loadingSpinner').classList.add('hidden');

                    // Tunggu gambar dimuat untuk mendapatkan dimensi
                    uploadedImageElement.onload = () => {
                        // Pastikan canvas berukuran sama dengan gambar
                        setCanvasSize();
                    };
                }
                reader.readAsDataURL(uploadedFile);
            }
        }

        function setCanvasSize() {
            // Dapatkan dimensi gambar yang ditampilkan (penting untuk responsivitas)
            const imgWidth = uploadedImageElement.offsetWidth;
            const imgHeight = uploadedImageElement.offsetHeight;

            // Set dimensi canvas
            detectionCanvas.width = imgWidth;
            detectionCanvas.height = imgHeight;
            // Posisi canvas sudah diatur oleh CSS: absolute top-0 left-0
        }

        window.addEventListener('resize', () => {
            if (uploadedFile) {
                setCanvasSize();
                // Jika sudah diproses, gambar ulang kotak deteksi
                if (currentDetections.length > 0) {
                    drawBoundingBoxes(currentDetections);
                }
            }
        });

        // Fungsi untuk memilih hasil Lion atau Cheetah (Simulasi)
        function generateSingleResult() {
            // Memilih secara acak antara Lion atau Cheetah
            const isLion = Math.random() < 0.5;
            const className = isLion ? "Lion" : "Cheetah";
            const color = isLion ? '#FF7F50' : '#ADD8E6'; // Coral atau Light Blue
            const confidence = (0.95 + Math.random() * 0.05); // Kepercayaan tinggi

            return {
                classification: className,
                detections: [
                    { 
                        class: className, 
                        color: color, 
                        // Bounding box mock di tengah
                        box: [250 + Math.random() * 50, 200 + Math.random() * 50, 750 + Math.random() * 50, 600 + Math.random() * 50], 
                        confidence: confidence 
                    }
                ]
            };
        }

        // 2. Pemrosesan Gambar (Menggantikan Panggilan API Model Sebenarnya)
        function processImage() {
            if (!uploadedFile) {
                showMessage("Silakan unggah gambar terlebih dahulu.");
                return;
            }

            // Generate hasil tunggal baru
            currentMockResult = generateSingleResult();

            // Tampilkan loading state
            processButton.disabled = true;
            document.getElementById('buttonText').classList.add('hidden');
            document.getElementById('loadingSpinner').classList.remove('hidden');
            detectionListElement.innerHTML = '<li class="text-center text-sm text-pink-400 py-2">Model sedang menganalisis gambar...</li>';
            classificationResultElement.textContent = 'Sedang Dihitung...';
            classificationResultElement.classList.remove('bg-fuchsia-800/50', 'border-fuchsia-500/80');
            classificationResultElement.classList.add('bg-gray-700/30', 'border-gray-600');
            
            currentDetections = currentMockResult.detections;

            // Simulasi waktu pemrosesan 3 detik
            setTimeout(() => {
                // Sembunyikan loading state
                processButton.disabled = false;
                document.getElementById('buttonText').classList.remove('hidden');
                document.getElementById('loadingSpinner').classList.add('hidden');

                // Tampilkan hasil Klasifikasi (H5 Model) - Tanpa Persentase
                classificationResultElement.textContent = currentMockResult.classification; 
                classificationResultElement.classList.remove('bg-gray-700/30', 'border-gray-600');
                classificationResultElement.classList.add('bg-fuchsia-800/50', 'border-fuchsia-500/80');

                // Tampilkan hasil Deteksi Objek (YOLOv8)
                displayDetectionList(currentDetections);
                
                // Gambar bounding boxes di canvas
                drawBoundingBoxes(currentDetections);

                showMessage("🎉 Pemrosesan Selesai! Hasil model telah ditampilkan.");

            }, 3000); // Simulasi waktu proses
        }

        // 3. Tampilkan Daftar Deteksi (YOLOv8)
        function displayDetectionList(detections) {
            detectionListElement.innerHTML = '';
            if (detections.length === 0) {
                detectionListElement.innerHTML = '<li class="text-sm text-gray-500 py-2">Tidak ada objek Lion atau Cheetah yang terdeteksi.</li>';
                return;
            }

            // Karena kita tahu hanya ada satu (Lion atau Cheetah)
            const detection = detections[0]; 
            const li = document.createElement('li');
            li.className = 'flex justify-between items-center p-3 bg-gray-700/50 rounded-lg border-l-4';
            li.style.borderColor = detection.color;

            li.innerHTML = `
                <span class="font-bold text-white">${detection.class}</span>
                <span class="text-xs font-mono px-3 py-1 rounded-full text-black" style="background-color: ${detection.color};">
                    ${(detection.confidence * 100).toFixed(1)}%
                </span>
            `;
            detectionListElement.appendChild(li);
        }

        // 4. Gambar Bounding Boxes di Canvas
        function drawBoundingBoxes(detections) {
            if (!uploadedImageElement.complete) {
                uploadedImageElement.onload = () => drawBoundingBoxes(detections);
                return;
            }
            
            setCanvasSize();
            const ctx = detectionCanvas.getContext('2d');
            const canvasWidth = detectionCanvas.width; 
            const canvasHeight = detectionCanvas.height;

            ctx.clearRect(0, 0, canvasWidth, canvasHeight); // Bersihkan canvas

            detections.forEach(detection => {
                // Koordinat kotak: [x_min, y_min, x_max, y_max] (dari 0 hingga 1000 dalam mock data)
                const x = detection.box[0] * (canvasWidth / 1000);
                const y = detection.box[1] * (canvasHeight / 1000);
                const w = (detection.box[2] - detection.box[0]) * (canvasWidth / 1000);
                const h = (detection.box[3] - detection.box[1]) * (canvasHeight / 1000);

                // Gambar Bounding Box
                ctx.strokeStyle = detection.color;
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);

                // Gambar Label
                const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
                ctx.font = '700 14px Inter'; 
                
                // Ukur teks untuk kotak latar belakang
                const textMetrics = ctx.measureText(label);
                const textWidth = textMetrics.width;
                const textHeight = 20;

                // Kotak latar belakang label
                ctx.fillStyle = detection.color;
                ctx.fillRect(x - 1, y - textHeight, textWidth + 10, textHeight);

                // Teks Label
                ctx.fillStyle = '#0d1117'; 
                ctx.fillText(label, x + 4, y - 5);
            });
        }
    </script>
</body>
</html>
"""

# Tanamkan konten HTML ke Streamlit
components.html(html_content, height=1000, scrolling=True)
