import cv2
import mediapipe as mp
import time
import time
import torch

use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"
cap = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
FaceMesh = mpFaceMesh.FaceMesh()

ptime = 0

def overlay_filter(frame, filter_img, bbox):
    x, y, w, h = bbox
    filter_img = cv2.resize(filter_img, (w, h))
    for i in range(h):
        for j in range(w):
            if y + i >= frame.shape[0] or x + j >= frame.shape[1]:
                continue
            alpha = filter_img[i, j, 3] / 255.0  
            if alpha > 0:
                for c in range(3):  
                    frame[y + i, x + j, c] = (1 - alpha) * frame[y + i, x + j, c] + alpha * filter_img[i, j, c]

def apply_filter_to_face(frame, left_eye, right_eye):
    filter_img = cv2.imread('filter.png', cv2.IMREAD_UNCHANGED)  
    
    if filter_img is None:
        print("Error: Unable to load the filter image.")
        return frame  

    filter_width = int(abs(left_eye.x - right_eye.x) * frame.shape[1])
    filter_height = int(abs(left_eye.y - right_eye.y) * frame.shape[0])

    filter_img = cv2.resize(filter_img, (filter_width, filter_height))
    
    x_pos = int(left_eye.x * frame.shape[1] - filter_width / 2)
    y_pos = int(left_eye.y * frame.shape[0] - filter_height / 2)
    
    overlay_filter(frame, filter_img, (x_pos, y_pos, filter_width, filter_height))

while True:
    start_time = time.time()
    istrue, frame = cap.read()
    if not istrue:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(frameRGB)

    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime
    cv2.putText(frame, f"fps: {int(fps)}", (28, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mpFaceMesh.FACEMESH_TESSELATION)

            left_eye = landmarks.landmark[33]  
            right_eye = landmarks.landmark[133]

            apply_filter_to_face(frame, left_eye, right_eye)

    cv2.imshow("Face Mesh", frame)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"[{device}] FPS: {fps:.2f}")
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
