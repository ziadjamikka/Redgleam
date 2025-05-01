import cv2
import mediapipe as mp
import numpy as np
import time
import torch

# Check device
use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"

# Load dog filter
dog_filter = cv2.imread("filters/Photos_IOS/Dog.png", cv2.IMREAD_UNCHANGED)
if dog_filter is None:
    raise ValueError("Could not load the filter image. Check the file path.")

if dog_filter.shape[2] == 3:
    dog_filter = cv2.cvtColor(dog_filter, cv2.COLOR_BGR2BGRA)

def rotate_and_scale(image, angle, scale=1.0):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bg = background.copy()

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
    if x + w > bg_w:
        overlay = overlay[:, :bg_w - x]
        w = bg_w - x
    if y + h > bg_h:
        overlay = overlay[:bg_h - y, :]
        h = bg_h - y

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]
    else:
        alpha = np.ones((h, w))
        overlay_rgb = overlay

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * overlay_rgb[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c]
        )

    return bg

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]

            face_width = int(abs(right_cheek.x - left_cheek.x) * w * 2.2) 
            face_height = int(abs(chin.y - forehead.y) * h * 3.5)

            center_x = int((left_cheek.x + right_cheek.x) * w / 2)
            center_y = int((forehead.y + chin.y) * h / 2)

            filter_aspect_ratio = dog_filter.shape[1] / dog_filter.shape[0]
            new_height = face_height 
            new_width = int(new_height * filter_aspect_ratio)

            if new_width > face_width:
                new_width = face_width
                new_height = int(new_width / filter_aspect_ratio)

            top_left_x = int(center_x - new_width / 2)+15
            top_left_y = int(center_y - new_height / 2) - int(0.2 * new_height) + 78

            new_width = max(10, new_width)
            new_height = max(10, new_height)

            rotation_angle = 10
            scale_factor = 1.1
            dog_filter_transformed = rotate_and_scale(dog_filter, rotation_angle, scale_factor)

            try:
                frame = overlay_transparent(frame, dog_filter_transformed, top_left_x, top_left_y, (new_width+70, new_height+180))
            except Exception as e:
                print(f"Error applying filter: {e}")
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    print(f"[{device}] FPS: {fps:.2f}")

    cv2.imshow("Dog Filter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
