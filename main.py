import cv2
import mediapipe as mp
import threading
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load trained models
eye_model = load_model("trained_models/eye_state_model.h5")
yawn_model = load_model("trained_models/yawn_detection_model.h5")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# Indices of important facial landmarks
LEFT_EYE_INDICES = [34, 245, 69, 101]
RIGHT_EYE_INDICES = [264, 465, 299, 330]
MOUTH_INDICES = [152, 164, 212, 432]   # Mouth

# Global variables: current frame and last detection results
frame = None
last_detections = {}

# Webcam capture
cap = cv2.VideoCapture(0)

# Preprocessing: convert to grayscale, resize to 64x64, normalize and reshape for the model.
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype("float32") / 255.0
    # Expand dims to match model input shape: (1, 64, 64, 1)
    return np.expand_dims(normalized, axis=(0, -1))

# Helper function that computes a bounding box from landmarks,
# preprocesses the cropped region, and runs the prediction.
# The 'invert' parameter allows us to reverse the prediction logic (used for eye model).
def process_and_get_detection(face_landmarks, landmarks_indices, model,
                              pos_label, neg_label, width, height, threshold=0.5, invert=False):
    # Compute bounding box coordinates from landmarks
    x_coords = [int(face_landmarks.landmark[i].x * width) for i in landmarks_indices]
    y_coords = [int(face_landmarks.landmark[i].y * height) for i in landmarks_indices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Extract a copy of the cropped region (to avoid drawing artifacts)
    crop = frame.copy()[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    # Preprocess crop and get prediction (assumed to be a single scalar)
    preprocessed = preprocess_crop(crop)
    pred = model.predict(preprocessed)[0][0]

    # For eyes, we invert the logic if invert=True.
    # For eyes: if pred > threshold then Eyes Open (green), else Eyes Closed (red).
    # For mouth (invert=False): if pred > threshold then Yawning (red), else Mouth Closed (green).
    if not invert:
        if pred > threshold:
            label = neg_label
            color = (0, 0, 255)  # red
        else:
            label = pos_label
            color = (0, 255, 0)  # green
    else:
        if pred > threshold:
            label = pos_label
            color = (0, 255, 0)  # green
        else:
            label = neg_label
            color = (0, 0, 255)  # red

    # Return bounding box coordinates, label and color.
    return (x_min, y_min, x_max, y_max, label, color)

# The processing thread continuously reads frames and runs detection.
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

            # Flip the frame horizontally for natural view and update global frame
            current_frame = cv2.flip(current_frame, 1)
            frame = current_frame.copy()
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            detections = {}
            if results.multi_face_landmarks:
                # Process the first detected face (max_num_faces is set to 1)
                for face_landmarks in results.multi_face_landmarks:
                    # For eyes, use invert=True
                    left_eye_det = process_and_get_detection(face_landmarks, LEFT_EYE_INDICES,
                                                             eye_model, "Eyes Open", "Eyes Closed",
                                                             width, height, invert=True)
                    right_eye_det = process_and_get_detection(face_landmarks, RIGHT_EYE_INDICES,
                                                              eye_model, "Eyes Open", "Eyes Closed",
                                                              width, height, invert=True)
                    # For mouth, use invert=False (default)
                    mouth_det = process_and_get_detection(face_landmarks, MOUTH_INDICES,
                                                          yawn_model, "Mouth Closed", "Yawning",
                                                          width, height)
                    if left_eye_det:
                        detections["left_eye"] = left_eye_det
                    if right_eye_det:
                        detections["right_eye"] = right_eye_det
                    if mouth_det:
                        detections["mouth"] = mouth_det

            # Update global last_detections if new detections are available.
            if detections:
                last_detections = detections

            # Add a delay to reduce the frame rate and lower CPU usage
            time.sleep(0.05)

# Start the frame processing thread
thread = threading.Thread(target=process_frame)
thread.daemon = True
thread.start()

# Main loop: display the current frame and overlay the detection boxes.
while True:
    if frame is not None:
        display_frame = frame.copy()
        # Overlay each detection box from the last known detections
        for key, (x_min, y_min, x_max, y_max, label, color) in last_detections.items():
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(display_frame, label, (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Real-time Face, Eyes, and Mouth Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
