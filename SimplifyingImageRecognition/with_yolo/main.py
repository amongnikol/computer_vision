from ultralytics import YOLO
import cv2

# Load YOLOv8 (pretrained on COCO)
model = YOLO("yolov8n.pt")  # atau "yolov5s.pt" jika ingin YOLOv5

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Gunakan CAP_DSHOW di Windows

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan prediksi
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Ambil frame dengan bounding box dari hasil prediksi
    annotated_frame = results[0].plot()

    # Tampilkan ke layar
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
