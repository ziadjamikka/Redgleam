import cv2
import mediapipe as mp
import numpy as np
import time
import torch

use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"

hat_img = cv2.imread("filters/Photos_IOS/hat-removebg-preview.png", cv2.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            forehead_landmark = landmarks.landmark[10]
            x = int(forehead_landmark.x * w)
            y = int(forehead_landmark.y * h)

            scale = int(w * 0.6) 
            hat_resized = cv2.resize(hat_img, (scale+1, scale+1))

            hat_h, hat_w, _ = hat_resized.shape
            x1 = x - hat_w // 2
            y1 = y - hat_h + 170  


            for i in range(hat_h):
                for j in range(hat_w):
                    if 0 <= y1 + i < h and 0 <= x1 + j < w:
                        alpha = hat_resized[i, j, 3] / 255.0
                        if alpha > 0:
                            frame[y1 + i, x1 + j] = (
                                alpha * hat_resized[i, j, :3] + (1 - alpha) * frame[y1 + i, x1 + j]
                            )
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"[{device}] FPS: {fps:.2f}")
    cv2.imshow('Hat Filter', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
