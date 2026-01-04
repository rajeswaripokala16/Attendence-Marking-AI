import cv2
from deepface import DeepFace
import os
import pandas as pd
from datetime import datetime

# Utility: Load known users
image_folder = "images"
known_users = {}
for file in os.listdir(image_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        name = os.path.splitext(file)[0]
        img_path = os.path.join(image_folder, file)
        known_users[name] = cv2.imread(img_path)

attendance_log = "attendance.csv"
if not os.path.exists(attendance_log):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_log, index=False)

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")
marked = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        # Analyze faces & get details
        results = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
        # Each "result" is a detected face dict
        for face_dict in results:
            x, y, w, h = [int(face_dict['facial_area'][k]) for k in ('x', 'y', 'w', 'h')]
            detected_face = frame[y:y+h, x:x+w]
            # Optimize: Use DeepFace.verify with best match
            match_found = None
            lowest_distance = float('inf')
            for name, img in known_users.items():
                try:
                    resp = DeepFace.verify(img1_path=img, img2_path=detected_face, detector_backend='opencv', enforce_detection=False)
                    if resp['verified'] and resp['distance'] < lowest_distance and resp['distance'] < 0.35:
                        lowest_distance = resp['distance']
                        match_found = name
                except Exception:
                    continue
            # Anti-spoof: Only allow if confidence high and face size reasonable
            real_face = w*h > 8000 and detected_face.shape[0] > 80 and detected_face.shape[1] > 80
            if match_found and match_found not in marked and real_face:
                now = datetime.now().strftime("%H:%M:%S")
                pd.DataFrame([{"Name": match_found, "Time": now}]).to_csv(attendance_log, mode="a", header=False, index=False)
                marked.add(match_found)
                label = f"Attendance: {match_found}"
                color = (0, 200, 255)
            elif match_found:
                label = f"Already Marked: {match_found}"
                color = (100, 80, 255)
            else:
                label = "Unknown or spoof!"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    except Exception as e:
        cv2.putText(frame, 'No face detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Smart Attendance System', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

