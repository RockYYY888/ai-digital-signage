# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import time
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import queue
from CV.model import EmotionClassifier, FaceAttributeModel, predict2
from CV.UTKFaceDataset import age_group_transform, gender_mapping, race_mapping, emotion_mapping
from Server.data_interface import secondary_screen_signal_queue, frame_queue
from util import get_resource_path

# Device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""torch.device: The device (CUDA GPU or CPU) used for model inference."""

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""transforms.Compose: Preprocessing pipeline for resizing, tensor conversion, and normalization of images."""

# Load models
model_emotion = EmotionClassifier(num_classes=4).to(device)
"""EmotionClassifier: Model for predicting emotions, loaded onto the selected device."""
model_demographic = FaceAttributeModel(num_age_classes=4, num_gender_classes=2, num_race_classes=5).to(device)
"""FaceAttributeModel: Model for predicting demographic attributes, loaded onto the selected device."""
checkpoint_path = get_resource_path('CV/best_face_attribute_model.pth')
checkpoint_path2 = get_resource_path('CV/best_project_model.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint2 = torch.load(checkpoint_path2, map_location=device)
model_demographic.load_state_dict(checkpoint['model_state_dict'])
model_emotion.load_state_dict(checkpoint2['model_state_dict'])

# Initialize YOLO face detector
face_detector = YOLO(get_resource_path("CV/yolov8l-face.pt"), verbose=False)
"""YOLO: Pre-trained YOLO model for face detection, initialized with a specific weights file."""


def analyze_frame(frame):
    """Analyze a single video frame to detect faces and predict demographic and emotional attributes.

    This function uses a YOLO face detector to identify faces in the frame, crops the detected face,
    and predicts age, gender, race, and emotion using pre-trained models.

    Args:
        frame (numpy.ndarray): The input frame in BGR format from OpenCV.

    Returns:
        tuple: A tuple containing:
            - combined_prediction (tuple or None): (age_label, gender_label, race_label, emotion_label) or None if no face is detected.
            - pil_image (PIL.Image or None): The cropped face image or None if no face is detected.
    """
    # Face detection
    results = face_detector(frame, conf=0.86)

    if not results or len(results[0].boxes.xyxy) == 0:
        print("[CV] No face detected at this frame.")
        return None, None

    # Process the first detected face
    box = results[0].boxes.xyxy[0]
    padding = 35
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    cropped_image = frame[y1:y2, x1:x2]

    # Convert to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # Predict demographics
    age_label, gender_label, race_label = predict_demographics(model_demographic, pil_image)

    # Predict emotion
    emotion_pred = predict2(model_emotion, pil_image)
    emotion_label = emotion_mapping.get(emotion_pred, "Unknown")
    combined_prediction = (age_label, gender_label, race_label, emotion_label)
    print(f"[CV] Predicted Demographics: {combined_prediction}")

    secondary_screen_signal_queue.put(combined_prediction)
    return combined_prediction, pil_image


def predict_demographics(model, image):
    """Predict demographic attributes (age, gender, race) from a face image.

    Args:
        model (FaceAttributeModel): The pre-trained model for demographic prediction.
        image (PIL.Image): The input face image in RGB format.

    Returns:
        tuple: A tuple containing:
            - age_label (str): Predicted age group.
            - gender_label (str): Predicted gender.
            - race_label (str): Predicted race/ethnicity.
    """
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, age_pred = torch.max(outputs['age'], 1)
        _, gender_pred = torch.max(outputs['gender'], 1)
        _, race_pred = torch.max(outputs['race'], 1)

    age_group = age_pred.item()
    gender = gender_pred.item()
    race = race_pred.item()

    age_label = age_group_transform(age_group)
    gender_label = [k for k, v in gender_mapping.items() if v == gender][0]
    race_label = [k for k, v in race_mapping.items() if v == race][0]

    return age_label, gender_label, race_label


def cv_thread_func(cap, detected_face_queue, face_detection_active):
    """Run computer vision processing in a thread to detect and analyze faces from a video stream.

    This function continuously captures frames from a video source, analyzes them for faces at
    specified intervals, and queues the results when a face is detected.

    Args:
        cap (cv2.VideoCapture): The video capture object for the webcam.
        detected_face_queue (queue.Queue): Queue to store detected face data (image and prediction).
        face_detection_active (threading.Event): Event to control when face detection is active.

    Raises:
        SystemExit: If the webcam fails to capture frames or an unrecoverable error occurs.
    """
    detection_interval = 2
    last_detection_time = time.time()

    if not cap.isOpened():
        print("[CV] Failed to open webcam.")
        return

    try:
        while True:
            if not face_detection_active.is_set():
                time.sleep(2)
                continue

            ret, frame = cap.read()
            if not ret or not cap.isOpened():
                print("[CV] Failed to capture frame or camera closed. Exiting loop.")
                break

            current_time = time.time()
            time_since_last = current_time - last_detection_time
            if time_since_last >= detection_interval:
                try:
                    prediction, cropped_image = analyze_frame(frame)
                    if prediction:
                        try:
                            cropped_image_bgr = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                            frame_queue.put_nowait(cropped_image_bgr)
                            detected_face_queue.put_nowait((cropped_image_bgr, prediction))
                            face_detection_active.clear()  # Pause face detection
                        except queue.Full:
                            print("[CV] Queue full, skipping frame.")
                except Exception as e:
                    print(f"[CV] Error in analyze_frame or subsequent processing: {e}")
                    continue
                last_detection_time = time.time()
                current_time = time.time()

            time.sleep(1.0)

    except Exception as e:
        print(f"[CV] Unexpected error in thread: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main entry point for testing OpenCV build information."""
    print(cv2.getBuildInformation())