import cv2
import numpy as np
import mediapipe as mp
import time
import torch

use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Lip indices
LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270,
    409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Colors and opacity
lip_color = (0, 0, 255)  
lip_opacity = 0.6

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            overlay = frame.copy()

            lips = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in LIPS_IDX]
            if lips:
                lips_np = np.array(lips, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [lips_np], lip_color)

            # ===== Blending Lipstick with Frame =====
            frame = cv2.addWeighted(overlay, lip_opacity, frame, 1 - lip_opacity, 0)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"[{device}] FPS: {fps:.2f}")
    cv2.imshow('Makeup Filter - Lipstick & Eyelashes', frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()