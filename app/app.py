import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Load model YOLO
@st.cache_resource  # Cache model biar gak reload tiap run
def load_model():
    model_path = "best.pt"  # Atau upload model
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error("Model best.pt tidak ditemukan di folder app!")
        st.stop()

model = load_model()

st.title("🧾 Deteksi Kesegaran Daging dengan YOLOv8")
st.markdown("Upload gambar daging untuk cek fresh/half/rotten")

# Sidebar untuk config
st.sidebar.header("Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Upload file
uploaded_file = st.file_uploader("Pilih gambar daging...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load & preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption="Upload", use_column_width=True)
    
    # Convert ke OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Inference YOLO
    results = model(img_cv, conf=confidence, verbose=False)
    
    # Plot results
    res_img = results[0].plot()  # Bounding boxes + labels
    
    st.image(res_img, caption="Deteksi Kesegaran", use_column_width=True)
    
    # Extract info
    boxes = results[0].boxes
    if boxes is not None:
        names = results[0].names
        st.subheader("Hasil Deteksi")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            st.write(f"**{label}** (confidence: {conf:.2f})")
    else:
        st.warning("Tidak ada objek terdeteksi!")

# Info
with st.expander("Tentang App"):
    st.info("""
    - **Model**: YOLOv8 custom-trained kesegaran daging (fresh/half/rotten).
    - **Dataset**: Roboflow (deteksi-kesegaran-daging + meat-freshness).
    - **Repo**: [git-yolo-daging](https://github.com/bot-lf/git-yolo-daging)
    """)

st.markdown("---")
st.caption("Skripsi Computer Vision 2026 © tvrumah")
