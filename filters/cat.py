import cv2
import mediapipe as mp
import numpy as np
import time
import torch

use_gpu = torch.cuda.is_available()
device = "GPU" if use_gpu else "CPU"

dog_filter = cv2.imread("filters/Photos_IOS/cat.png", cv2.IMREAD_UNCHANGED)
if dog_filter is None:
    raise ValueError("Could not load the filter image. Check the file path.")
if dog_filter.shape[2] == 3:
    dog_filter = cv2.cvtColor(dog_filter, cv2.COLOR_BGR2BGRA)

def rotate_and_scale(image, angle, scale=1.0):
    h, w = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    return cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, background.shape[1] - x)
    h = min(h, background.shape[0] - y)

    if w <= 0 or h <= 0:
        return background

    overlay_crop = overlay[:h, :w]
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha * overlay_rgb[:, :, c] +
                                           (1 - alpha) * background[y:y+h, x:x+w, c])
    return background

mp_face_mesh = mp.solutions.face_mesh

cpu_times = []
gpu_times = []

use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        start_time = time.time()

        if use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = gpu_rgb.download()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[152]
                left_cheek = face_landmarks.landmark[234]
                right_cheek = face_landmarks.landmark[454]

                face_width = int(abs(right_cheek.x - left_cheek.x) * w * 2.2)
                face_height = int(abs(chin.y - forehead.y) * h * 3.5)
                center_x = int((left_cheek.x + right_cheek.x) * w / 2)
                center_y = int((forehead.y + chin.y) * h / 2)

                aspect = dog_filter.shape[1] / dog_filter.shape[0]
                new_height = face_height
                new_width = int(new_height * aspect)

                if new_width > face_width:
                    new_width = face_width
                    new_height = int(new_width / aspect)

                top_left_x = int(center_x - new_width / 2) - 80
                top_left_y = int(center_y - new_height / 2) - int(0.2 * new_height) - 30

                rotation_angle = 22
                scale_factor = 1.1
                transformed = rotate_and_scale(dog_filter, rotation_angle, scale_factor)

                frame = overlay_transparent(frame, transformed, top_left_x + 50, top_left_y,
                                            (new_width + 115, new_height + 120))

        total_time = time.time() - start_time
        if use_gpu:
            gpu_times.append(total_time)
        else:
            cpu_times.append(total_time)

        cv2.imshow("Dog Filter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"[{device}] FPS: {fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()

avg_cpu = np.mean(cpu_times) if cpu_times else 0
avg_gpu = np.mean(gpu_times) if gpu_times else 0

print(f"\nAverage Frame Time - CPU: {avg_cpu:.4f} sec")
if use_gpu:
    print(f"Average Frame Time - GPU: {avg_gpu:.4f} sec")
else:
    print("GPU not available.")
