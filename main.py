import cv2
import mediapipe as mp
import threading
import numpy as np
import time
import pygame
from tensorflow.keras.models import load_model

# Initialize pygame mixer for playing mp3 sounds
pygame.mixer.init()
# Load alert sound files (ensure these files exist in the folder)
eye_alert_sound = pygame.mixer.Sound("sounds/eye_alert.wav")
yawn_alert_sound = pygame.mixer.Sound("sounds/yawn_alert.wav")

# Load trained models
eye_model = load_model("trained_models/eye_state_model.h5")
yawn_model = load_model("trained_models/yawn_detection_model.h5")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# Indices of important facial landmarks
LEFT_EYE_INDICES = [139, 245, 69, 101]
RIGHT_EYE_INDICES = [368, 465, 299, 330]
MOUTH_INDICES = [199, 164, 212, 432]  # Mouth

# Thread-safe globals using locks
frame_lock = threading.Lock()
last_detections_lock = threading.Lock()
frame = None
last_detections = {}

# Counters and alert flags
eye_closed_counter = 0
yawning_counter = 0
yawn_events = []
eye_alert_triggered = False
# Note: We remove the persistent yawning alert flag so that events can trigger repeatedly

# Initialize webcam capture and set a lower resolution for performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def process_and_get_detection(face_landmarks, landmarks_indices, model,
                              pos_label, neg_label, width, height, threshold=0.5, invert=False):
    # Compute bounding box using given landmarks
    x_coords = [int(face_landmarks.landmark[i].x * width) for i in landmarks_indices]
    y_coords = [int(face_landmarks.landmark[i].y * height) for i in landmarks_indices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Safely copy the current frame
    with frame_lock:
        if frame is None:
            return None
        crop = frame[y_min:y_max, x_min:x_max].copy()
    if crop.size == 0:
        return None

    preprocessed = preprocess_crop(crop)
    pred = model.predict(preprocessed, verbose=0)[0][0]

    # Determine label and color based on prediction and invert flag
    if not invert:
        if pred > threshold:
            label = neg_label
            color = (0, 0, 255)  # red for alert
        else:
            label = pos_label
            color = (0, 255, 0)  # green for safe
    else:
        if pred > threshold:
            label = pos_label
            color = (0, 255, 0)
        else:
            label = neg_label
            color = (0, 0, 255)
    return (x_min, y_min, x_max, y_max, label, color)

def process_frame():
    global frame, last_detections
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            # Flip the frame for a natural view
            current_frame = cv2.flip(current_frame, 1)
            with frame_lock:
                frame = current_frame.copy()

            height, width, _ = current_frame.shape
            rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            detections = {}
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_det = process_and_get_detection(face_landmarks, LEFT_EYE_INDICES,
                                                             eye_model, "Eyes Open", "Eyes Closed",
                                                             width, height, invert=True)
                    right_eye_det = process_and_get_detection(face_landmarks, RIGHT_EYE_INDICES,
                                                              eye_model, "Eyes Open", "Eyes Closed",
                                                              width, height, invert=True)
                    mouth_det = process_and_get_detection(face_landmarks, MOUTH_INDICES,
                                                          yawn_model, "Mouth Closed", "Yawning",
                                                          width, height)
                    if left_eye_det:
                        detections["left_eye"] = left_eye_det
                    if right_eye_det:
                        detections["right_eye"] = right_eye_det
                    if mouth_det:
                        detections["mouth"] = mouth_det
            with last_detections_lock:
                if detections:
                    last_detections = detections

            # Short sleep to slightly ease CPU load
            time.sleep(0.01)

# Start detection in a separate thread
threading.Thread(target=process_frame, daemon=True).start()

# Main loop for display and counter updates
while True:
    with frame_lock:
        if frame is None:
            continue
        display_frame = frame.copy()

    with last_detections_lock:
        detections = last_detections.copy()

    # Update counters based on latest detections
    if "left_eye" in detections and "right_eye" in detections:
        if (detections["left_eye"][4] == "Eyes Closed" and
            detections["right_eye"][4] == "Eyes Closed"):
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0
            eye_alert_triggered = False

    if "mouth" in detections:
        if detections["mouth"][4] == "Yawning":
            yawning_counter += 1
        else:
            if yawning_counter >= 40:
                yawn_events.append(time.time())
            yawning_counter = 0

    # Keep only yawning events in the past 30 minutes (1800 seconds)
    current_time = time.time()
    yawn_events = [t for t in yawn_events if current_time - t < 1800]

    # Trigger alert sounds if thresholds are exceeded
    if eye_closed_counter > 600 and not eye_alert_triggered:
        eye_alert_sound.play()
        eye_alert_triggered = True

    # If there are 3 or more yawning events, trigger the yawning alert
    if len(yawn_events) >= 3:
        yawn_alert_sound.play()
        # Reset the yawning events so that the alert can be triggered again later
        yawn_events = []

    # Draw detection boxes and overlay counters
    for key, (x_min, y_min, x_max, y_max, label, color) in detections.items():
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(display_frame, label, (x_min, max(y_min - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, f"Eyes Closed Count: {eye_closed_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Yawning Counter: {yawning_counter}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Yawning Events: {len(yawn_events)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Real-time Face, Eyes, and Mouth Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
