from ultralytics import YOLO

# 1. Load model YOLO11 (nano biar ringan)
model = YOLO("yolo11n.pt")  # pertama kali jalan, file ini akan otomatis di-download

# 2. Train pakai dataset daging dari Roboflow
results = model.train(
    data="dataset/data.yaml",  # pastikan path ini benar
    epochs=15,                 # bisa kamu ubah nanti (20 dulu kalau mau cepat)
    imgsz=640,
    batch=4,
    name="daging-freshness-yolo11",
    device="cpu"               # ganti "0" kalau pakai GPU CUDA
)

print("Training selesai.")
