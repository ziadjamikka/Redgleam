import cv2
import mediapipe as mp
import numpy as np
import time
import torch

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"

def resize_for_processing(image, scale=0.5):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)

def draw_glasses(image, left_eye, right_eye, eye_distance, face_width, forehead, chin, iw, ih):
    glasses_width = int(face_width * 0.75)
    frame_thickness = 2
    lens_radius = int(face_width * 0.11)
    lens_radius_ = int(face_width * 0.22)
    face_height = int(abs(forehead[1] - chin[1]) * 1.1)

    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2
    y_offset = int(face_height * 0.05)

    bridge_length = int(face_width * 0.08)

    cv2.line(image,
             (center_x - bridge_length // 2, center_y + y_offset),
             (center_x + bridge_length // 2, center_y + y_offset),
             (0, 0, 0), frame_thickness)

    left_lens_center = (center_x - glasses_width // 2 + lens_radius_, center_y + y_offset)
    right_lens_center = (center_x + glasses_width // 2 - lens_radius_, center_y + y_offset)

    cv2.circle(image, left_lens_center, lens_radius, (0, 0, 0), frame_thickness)
    cv2.circle(image, right_lens_center, lens_radius, (0, 0, 0), frame_thickness)

def glasses_transformation(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

cap = cv2.VideoCapture(0)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    ih, iw = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    rgb_small = cv2.cvtColor(resize_for_processing(frame), cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_small)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            def scale_point(p):  
                return int(p.x * iw), int(p.y * ih)

            left_eye = scale_point(landmarks[33])
            right_eye = scale_point(landmarks[263])
            forehead = scale_point(landmarks[10])
            chin = scale_point(landmarks[152])

            eye_distance = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
            face_width = int(eye_distance * 2.5)

            draw_glasses(frame, left_eye, right_eye, eye_distance, face_width, forehead, chin, iw, ih)
            frame = glasses_transformation(frame)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"[{device}] FPS: {fps:.2f}")


    cv2.imshow("Smart Glasses Filter", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
