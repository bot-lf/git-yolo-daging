import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# path ke model (fresh-skripsi/models/best.pt)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("Deteksi Kesegaran Daging dengan YOLO")

uploaded_file = st.file_uploader("Upload gambar daging", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar asli", use_column_width=True)

    # simpan sementara dan prediksi
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        result = model.predict(source=tmp.name, imgsz=640, conf=0.25)[0]

    plotted = result.plot()  # BGR
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    st.image(plotted_rgb, caption="Hasil deteksi", use_column_width=True)

    if result.boxes is not None and len(result.boxes) > 0:
        st.subheader("Detail deteksi")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            st.write(f"- Kelas: {cls_name}, Confidence: {conf:.2f}")
    else:
        st.write("Tidak ada objek terdeteksi.")
else:
    st.write("Silakan upload gambar daging terlebih dahulu.")
