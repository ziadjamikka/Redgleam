import cv2
import dlib
import numpy as np

# تحميل كاشف الوجه وملامح الوجه
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("filters/shape_predictor_68_face_landmarks.dat")

# تحميل صورة الشنب (PNG بخلفية شفافة)
mustache_img = cv2.imread("filters/Photos_IOS/mustache.png", cv2.IMREAD_UNCHANGED)  # with alpha channel

# فتح الكاميرا
cap = cv2.VideoCapture(0)

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """دمج صورة PNG شفافة على خلفية"""
    bg = background.copy()

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[:2]

    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg  # خارج الإطار

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay

    for c in range(0, 3):
        bg[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                               alpha_bg * bg[y:y+h, x:x+w, c])

    return bg

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # نقاط الفم والأنف
        left = (landmarks.part(48).x, landmarks.part(48).y)
        right = (landmarks.part(54).x, landmarks.part(54).y)
        top = (landmarks.part(33).x, landmarks.part(33).y + 10)

        # حساب حجم وموقع الشنب
        mustache_width = right[0] - left[0]
        mustache_height = int(mustache_width * 0.4)

        x = left[0]
        y = top[1]

        # دمج صورة الشنب على الوجه
        frame = overlay_transparent(frame, mustache_img, x-23, y-23, (mustache_width+50, mustache_height+20))

    cv2.imshow("Live Mustache Filter", frame)

    # خروج عند الضغط على q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
