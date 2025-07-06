import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Load Model ===
model = load_model('models/best_model_CNN.h5')

# === Label Mapping ===
label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'} 

# === Open Webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocessing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              
    resized = cv2.resize(gray, (28, 28))                      
    normalized = resized.astype("float32") / 255.0           
    reshaped = normalized.reshape(1, 28, 28, 1)               

    # --- Predict ---
    pred = model.predict(reshaped)
    class_idx = np.argmax(pred)
    class_label = label_map[class_idx]

    # --- Tampilkan Hasil ---
    cv2.putText(frame, f"Prediksi: {class_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Rock Paper Scissors Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
