import cv2
import mediapipe as mp
import threading

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_INDICES = [34, 245, 69, 101]
RIGHT_EYE_INDICES = [264, 465, 299, 330]
MOUTH_INDICES = [152, 164, 212, 432]

frame = None

cap = cv2.VideoCapture(0)


def process_frame():
    global frame
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    def get_bounding_box(landmarks_indices):
                        x_coords = [int(face_landmarks.landmark[i].x * width) for i in landmarks_indices]
                        y_coords = [int(face_landmarks.landmark[i].y * height) for i in landmarks_indices]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        cropped_region = frame.copy()[y_min:y_max, x_min:x_max]

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        return cropped_region

                    left_eye_crop = get_bounding_box(LEFT_EYE_INDICES)
                    right_eye_crop = get_bounding_box(RIGHT_EYE_INDICES)
                    mouth_crop = get_bounding_box(MOUTH_INDICES)


thread = threading.Thread(target=process_frame)
thread.start()

while True:
    if frame is not None:
        cv2.imshow('Real-time Face, Eyes, and Mouth Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
