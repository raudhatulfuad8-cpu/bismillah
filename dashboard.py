import streamlit as st
import streamlit.components.v1 as components

# =========================
# 1️⃣ KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Visualisasi Model AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
.reportview-container .main .block-container {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 2️⃣ KONTEN HTML + JS
# =========================
html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visualisasi AI: Klasifikasi & Deteksi</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    body { font-family:'Inter',sans-serif; background:#0d1117; color:#c9d1d9; }
    #app { box-shadow:0 0 25px rgba(255,105,180,0.2); }
</style>
</head>

<body class="min-h-screen flex items-center justify-center p-4 md:p-8">
<div id="app" class="w-full max-w-4xl bg-gray-900/90 p-8 rounded-xl border border-fuchsia-700/50">

    <h1 class="text-3xl font-extrabold text-center mb-2 text-white">Deteksi Objek & Klasifikasi Gambar</h1>
    <p class="text-center mb-6 text-gray-400">Simulasi hasil model <span class='text-lime-400 font-bold'>YOLOv8</span> & <span class='text-pink-400 font-bold'>Classifier (.h5)</span></p>

    <!-- Upload Gambar -->
    <div class="border-2 border-dashed border-pink-500/50 rounded-xl p-6 mb-6 text-center bg-gray-800 hover:border-pink-400 cursor-pointer transition" onclick="document.getElementById('imageInput').click()">
        <input id="imageInput" type="file" accept="image/*" class="hidden" onchange="previewImage(event)">
        <p class="text-gray-300 font-semibold">Klik untuk unggah gambar (JPG/PNG)</p>
    </div>

    <!-- Area Hasil -->
    <div id="resultArea" class="hidden space-y-6">
        <div class="relative border border-gray-700 rounded-xl overflow-hidden">
            <img id="preview" class="w-full max-h-[400px] object-contain">
            <canvas id="canvas" class="absolute top-0 left-0 w-full h-full"></canvas>
        </div>

        <div class="grid md:grid-cols-2 gap-4">
            <div class="bg-gray-800 border border-fuchsia-600/50 p-4 rounded-xl">
                <h2 class="text-fuchsia-400 font-bold mb-2">Klasifikasi (.h5)</h2>
                <div id="classResult" class="text-center text-2xl bg-gray-700/30 p-3 rounded-lg">Menunggu...</div>
            </div>
            <div class="bg-gray-800 border border-lime-600/50 p-4 rounded-xl">
                <h2 class="text-lime-400 font-bold mb-2">Deteksi Objek (YOLOv8)</h2>
                <ul id="detectList" class="space-y-2 max-h-32 overflow-y-auto text-sm text-gray-300">Menunggu...</ul>
            </div>
        </div>

        <button id="processBtn" class="w-full py-3 mt-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-xl shadow-md hover:shadow-lg disabled:opacity-50" disabled onclick="processImage()">🚀 Proses Gambar</button>
    </div>

    <!-- Modal Notifikasi -->
    <div id="modal" class="hidden fixed inset-0 bg-black/70 items-center justify-center">
        <div class="bg-gray-800 border border-lime-500 p-6 rounded-xl text-center">
            <p id="modalText" class="text-lg font-bold text-lime-400 mb-3"></p>
            <button onclick="hideModal()" class="bg-lime-600 text-black px-4 py-2 rounded-lg">OK</button>
        </div>
    </div>
</div>

<script>
const scenarios = [
    { label:'Singa', color:'#FF4500', prob:'99.8%', box:[200,300,750,700] },
    { label:'Cheetah', color:'#FFD700', prob:'97.5%', box:[150,450,800,850] }
];
let imgEl, canvas, ctx, resultArea, btn;

document.addEventListener('DOMContentLoaded',()=>{
    imgEl=document.getElementById('preview');
    canvas=document.getElementById('canvas');
    ctx=canvas.getContext('2d');
    resultArea=document.getElementById('resultArea');
    btn=document.getElementById('processBtn');
});

function previewImage(e){
    const file=e.target.files[0];
    if(!file)return;
    const reader=new FileReader();
    reader.onload=(x)=>{
        imgEl.src=x.target.result;
        resultArea.classList.remove('hidden');
        document.getElementById('classResult').textContent='Menunggu...';
        document.getElementById('detectList').innerHTML='<li>Tekan tombol proses untuk menjalankan simulasi.</li>';
        btn.disabled=false;
    };
    reader.readAsDataURL(file);
}

function processImage(){
    btn.disabled=true;
    const choice=scenarios[Math.floor(Math.random()*scenarios.length)];
    setTimeout(()=>{
        // tampilkan hasil klasifikasi
        document.getElementById('classResult').textContent=`${choice.label} (${choice.prob})`;
        // tampilkan hasil deteksi
        document.getElementById('detectList').innerHTML=`
            <li class="flex justify-between p-2 bg-gray-700/50 rounded-md border-l-4" style="border-color:${choice.color}">
                <span>${choice.label}</span><span class="font-mono">${choice.prob}</span>
            </li>`;
        drawBox(choice);
        showModal("✅ Pemrosesan selesai! Hasil simulasi ditampilkan.");
        btn.disabled=false;
    },2500);
}

function drawBox(obj){
    setCanvas();
    ctx.strokeStyle=obj.color; ctx.lineWidth=3;
    const [x1,y1,x2,y2]=obj.box.map(v=>v*(canvas.width/1000));
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle=obj.color; ctx.font='bold 14px Inter';
    ctx.fillRect(x1,y1-22,ctx.measureText(obj.label).width+12,20);
    ctx.fillStyle='#000'; ctx.fillText(obj.label,x1+4,y1-8);
}

function setCanvas(){
    canvas.width=imgEl.offsetWidth;
    canvas.height=imgEl.offsetHeight;
}

function showModal(t){document.getElementById('modalText').textContent=t;document.getElementById('modal').classList.remove('hidden');document.getElementById('modal').classList.add('flex');}
function hideModal(){document.getElementById('modal').classList.add('hidden');}
</script>
</body>
</html>
"""

# =========================
# 3️⃣ TAMPILKAN DI STREAMLIT
# =========================
components.html(html_content, height=900, scrolling=True)

st.markdown("""
---
<p style="text-align:center;color:gray;font-size:13px">
Visualisasi ini merupakan simulasi. Hasil deteksi ditampilkan acak untuk <b>Singa</b> atau <b>Cheetah</b>.<br>
Ditanam menggunakan <code>st.components.v1.html()</code>.
</p>
""", unsafe_allow_html=True)
