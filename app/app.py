import os
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


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
        model = YOLO(model_path)
        return model, model_path
    else:
        st.error(f"Model '{model_filename}' tidak ditemukan! Dicari di: {model_path}")
        st.stop()


model, model_path = load_model()


# =========================
# 2. UI Utama
# =========================
st.set_page_config(
    page_title="Deteksi Kesegaran Daging",
    page_icon="🧾",
    layout="centered"
)

st.title("🧾 Deteksi Kesegaran Daging dengan YOLO11")
st.markdown(
    """
    Upload gambar daging untuk mendeteksi tingkat **kesegaran**:
    - `fresh`
    - `half`
    - `rotten`
    """
)

st.caption(f"Model path: `{model_path}`")

st.sidebar.header("⚙️ Pengaturan Deteksi")

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
    help="Threshold untuk Non-Maximum Suppression (penggabungan box yang overlap)."
)

show_raw_boxes = st.sidebar.checkbox(
    "Tampilkan data box mentah (debug)",
    value=False
)


# =========================
# 3. Upload & Deteksi
# =========================
uploaded_file = st.file_uploader(
    "Pilih gambar daging...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Kolom untuk before / after
    col1, col2 = st.columns(2)

    # Load & tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # Convert ke numpy (RGB) untuk YOLO
    img_np = np.array(image)

    # Inference YOLO11
    with st.spinner("Sedang mendeteksi kesegaran daging..."):
        results = model(
            img_np,
            conf=confidence,
            iou=iou_thres,
            verbose=False
        )

    # Gambar hasil deteksi (Ultralytics sudah handle drawing)
    res_img = results[0].plot()  # numpy array (RGB)
    with col2:
        st.subheader("Hasil Deteksi")
        st.image(res_img, use_container_width=True)

    # =========================
    # 3a. Detail hasil deteksi
    # =========================
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        st.subheader("📊 Ringkasan Deteksi")

        # Hitung jumlah per kelas
        class_counts = {}
        for box in boxes:
            cls = int(box.cls[0])
            label = names.get(cls, f"class_{cls}")
            class_counts[label] = class_counts.get(label, 0) + 1

        # Tabel ringkasan
        for label, count in class_counts.items():
            st.write(f"- **{label}**: {count} objek terdeteksi")

        st.markdown("---")
        st.subheader("🔎 Detail Deteksi per Objek")

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf_box = float(box.conf[0])
            label = names.get(cls, f"class_{cls}")
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            st.markdown(
                f"""
                **Objek {i+1}**  
                - Label: **{label}**  
                - Confidence: `{conf_box:.2f}`  
                - Bounding box: `[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]`
                """
            )

        if show_raw_boxes:
            st.markdown("---")
            st.subheader("🧪 Data Box Mentah (Debug)")
            st.write(boxes.data.cpu().numpy())

    else:
        st.warning(
            "Tidak ada objek terdeteksi! "
            "Coba turunkan threshold confidence/IoU atau gunakan gambar lain."
        )

else:
    st.info("Silakan upload gambar daging terlebih dahulu untuk memulai deteksi.")


# =========================
# 4. Info Tambahan
# =========================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.markdown(
        """
        - **Model**: YOLO11 custom-trained untuk kesegaran daging (fresh / half / rotten).  
        - **Framework**: Ultralytics YOLO11 + Streamlit.  
        - **Pengaturan**: Confidence & IoU threshold bisa diatur di sidebar sesuai kebutuhan eksperimen.  
        - **Output**: Visualisasi bounding box + ringkasan jumlah objek tiap kelas.
        """
    )

st.markdown("---")
st.caption("Skripsi Computer Vision · YOLO11 · Streamlit")
