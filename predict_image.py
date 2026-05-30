from ultralytics import YOLO

# استخدم المسار الصحيح لنموذجك
model = YOLO(r"C:\Users\user\Documents\TrashModel\runs\detect\runs\train\smart_waste_yolov8n-5\weights\best.pt")

# تشغيل inference وحفظ صورة النتيجة فقط
results = model.predict(
    source=r"C:\Users\user\Documents\TrashModel\test.jpg",
    conf=0.25,
    save=True,
    show=False,   # مهم: لا يفتح نافذة
    device="cpu"  # مهم: يتجنب كل مشاكل CUDA
)

print("Done. Check runs/detect/ folder for output image.")