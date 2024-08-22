import pose_media as pm
import numpy as np
import tensorflow as tf
import cv2
import time

threshold = 0.5  # Threshold untuk menentukan aksi
pTime = 0
cTime = 0
actions = np.array(["DRINKPIL", "NOACT"])  # Label aksi
pose = pm.mediapipe_pose()  # Inisialisasi pose Mediapipe
pt = pose.mp_holistic.Holistic()  # Gunakan holistic untuk mendeteksi pose
new_model = tf.keras.models.load_model('minum_obat_new.h5')  # Load model yang sudah dilatih
counter = 0  # Inisialisasi counter untuk menghitung aktivitas "minum obat"

sequence = []
sentence = []
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame tidak dapat diakses.")
        break
    
    try:
        frame, results = pose.mediapipe_detection(frame, pt)
    except Exception as e:
        print(f"Error saat deteksi mediapipe: {e}")
        continue

    # pose.draw_styled_landmarks(frame, results)
    keypoints = pose.extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-10:]  # Simpan hanya 30 frame terakhir

    if len(sequence) == 10:
        res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                    if actions[np.argmax(res)] == "DRINKPIL":
                        counter += 1  # Tambahkan counter jika aksi "DRINKPIL" terdeteksi
            else:
                sentence.append(actions[np.argmax(res)])
                if actions[np.argmax(res)] == "DRINKPIL":
                    counter += 1  # Tambahkan counter jika aksi "DRINKPIL" terdeteksi
        if len(sentence) > 1: 
            sentence = sentence[-1:]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(frame, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(sentence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Counter: " + str(counter), (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    resizeImage = cv2.resize(frame, (640, 480))
    cv2.imshow('Detect Action', resizeImage)

    if cv2.waitKey(1) & 0xFF == ord("q") or counter == 3:
        break

cap.release()
cv2.destroyAllWindows()
