import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# =========================
# 1. Load YOLO11 model
# =========================
@st.cache_resource
def load_model():
    # Folder tempat app.py berada (folder "app")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # GANTI nama file model di sini kalau beda
    model_filename = "best.pt"  # contoh: model YOLO11 custom kamu
    model_path = os.path.join(base_dir, model_filename)

    if os.path.exists(model_path):
        return YOLO(model_path), model_path
    else:
        st.error(f"Model '{model_filename}' tidak ditemukan! Dicari di: {model_path}")
        st.stop()

model, model_path = load_model()

# =========================
# 2. UI Utama
# =========================
st.title("🧾 Deteksi Kesegaran Daging dengan YOLO11")
st.markdown("Upload gambar daging untuk cek **fresh / half / rotten**")

st.caption(f"Model path: `{model_path}`")

# Sidebar untuk config
st.sidebar.header("Pengaturan Deteksi")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.35,
    step=0.05,
    help="Semakin tinggi, semakin sedikit deteksi tapi lebih yakin."
)

iou_thres = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
    help="Threshold untuk Non-Maximum Suppression."
)

show_raw_boxes = st.sidebar.checkbox(
    "Tampilkan data box mentah",
    value=False
)

# =========================
# 3. Upload & Deteksi
# =========================
uploaded_file = st.file_uploader(
    "Pilih gambar daging...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load & preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar upload", use_column_width=True)

    # Convert ke OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Inference YOLO11
    with st.spinner("Sedang mendeteksi kesegaran daging..."):
        results = model(
            img_cv,
            conf=confidence,
            iou=iou_thres,
            verbose=False
        )

    # Gambar hasil deteksi
    res_img_bgr = results[0].plot()              # numpy BGR
    res_img_rgb = cv2.cvtColor(res_img_bgr, cv2.COLOR_BGR2RGB)

    st.image(res_img_rgb, caption="Deteksi Kesegaran (YOLO11)", use_column_width=True)

    # Detail hasil
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        st.subheader("Hasil Deteksi (per objek)")

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf_box = float(box.conf[0])
            label = names.get(cls, f"class_{cls}")

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            st.write(
                f"**Objek {i+1}** — "
                f"Label: **{label}**, "
                f"Confidence: `{conf_box:.2f}`, "
                f"Box: `[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]`"
            )

        if show_raw_boxes:
            st.write("Raw boxes:")
            st.write(boxes.data.cpu().numpy())
    else:
        st.warning("Tidak ada objek terdeteksi! Coba turunkan threshold atau pakai gambar lain.")

else:
    st.info("Silakan upload gambar daging terlebih dahulu.")

# =========================
# 4. Info Tambahan
# =========================
with st.expander("Tentang App"):
    st.markdown(
        """
        - **Model**: YOLO11 custom-trained kesegaran daging (fresh / half / rotten).
        - **Framework**: Ultralytics YOLO11 + Streamlit.
        - **Pengaturan**: Confidence & IoU threshold bisa diubah di sidebar.
        """
    )

st.markdown("---")
st.caption("Skripsi Computer Vision fellix · YOLO11 · Streamlit")
