from ultralytics import YOLO

# استخدم المسار الصحيح لنموذجك
model = YOLO(r"C:\trash_detector.pt")


results = model.predict(
    source=r"test.jpg",
    conf=0.25,
    save=True,
    show=False, 
    device="cpu" 

print("Done. Check runs/detect/ folder for output image.")
