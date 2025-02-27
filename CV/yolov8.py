import threading
import time
import cv2
from ultralytics import YOLO
from model import *
from PIL import Image
import os
from LLM.LLM import *
import queue
import threading
from data_integration.server import *
from data_integration.data_interface import prediction_queue

# Initialize the webcam
cap = cv2.VideoCapture(0)
face_detector = YOLO("yolov8l-face.pt", verbose=False)
desired_size = (224, 224)
detection_interval = 2
last_detection_time = time.time()

# Create threading events for synchronization
model_event = threading.Event()
text_generation_done_event = threading.Event()  # New event for text generation synchronization

# Create a queue to limit the number of threads processing the frames

# Initialize threading events
model_event.set()  # Allow detection to start immediately

# Load the emotion and demographic models
model_emotion = EmotionClassifier(num_classes=4)
model_emotion = model_emotion.to(device)

model_demographic = FaceAttributeModel(num_age_classes, num_gender_classes, num_race_classes)
model_demographic = model_demographic.to(device)

# Load the pretrained models
checkpoint_path = 'best_face_attribute_model.pth'
checkpoint_path2 = 'best_project_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
checkpoint2 = torch.load(checkpoint_path2, map_location=torch.device('cpu'))
model_demographic.load_state_dict(checkpoint['model_state_dict'])
model_emotion.load_state_dict(checkpoint2['model_state_dict'])

# Webcam face detection function
def detect_faces_from_webcam():
    global model_event, text_generation_done_event, last_detection_time

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    print("Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            cv2.imshow("Webcam - Press ESC to Exit", frame)

            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                last_detection_time = current_time
                # Wait until the previous detection and text generation are complete
                if model_event.is_set():
                    threading.Thread(target=process_frame, args=(frame.copy(),)).start()

            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Frame processing function
def process_frame(initial_frame):
    global model_event, text_generation_done_event
    model_event.clear()

    # Perform histogram equalization (optional)
    frame_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    frame_eq = cv2.equalizeHist(frame_gray)
    frame_eq = cv2.cvtColor(frame_eq, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
    # Apply Gamma Correction (optional)
    gamma = 1.2  # Adjust gamma for lighting correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    frame_eq = cv2.LUT(frame_eq, table)

    # Perform face detection
    results = face_detector(frame_eq, conf=0.86)
    if len(results) == 0:
        print("[Info] No face detected at this timestamp.")
        model_event.set()  # Signal that the frame can be processed again
        return

    # Process the first detected face
    result = results[0]  # Assume we only need to process the first face found
    if len(result.boxes.xyxy) == 0:
        print("[Info] No face detected at this timestamp.")
        model_event.set()  # Signal that the frame can be processed again
        return

    print("[Info] Face detected and processing.")
    padding = 20
    box = result.boxes.xyxy[0]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(initial_frame.shape[1], x2 + padding)
    y2 = min(initial_frame.shape[0], y2 + padding)
    cropped_image = initial_frame[y1:y2, x1:x2]

    # Convert the cropped face to a PIL image for prediction
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # Predict demographic features (age, gender, race)
    age_label, gender_label, race_label = predict(model_demographic, pil_image)

    # Predict emotion
    emotion_pred = predict2(model_emotion, pil_image)
    emotion_label = emotion_mapping.get(emotion_pred, "Unknown")
    combined_prediction = (age_label, gender_label, race_label, emotion_label)

    # Ensure the 'faces' directory exists
    os.makedirs("faces", exist_ok=True)

    # Print the prediction
    print(f"Predicted Demographics: {combined_prediction}")
    print("Put combined_prediction into queue:", combined_prediction)

    # Put the prediction into the queue for data_interface build
    prediction_queue.put(combined_prediction)


    # Start a new thread for generating the advertisement text or further processing
    threading.Thread(target=generate_target_text_in_yolo, args=(combined_prediction,)).start()
    # Wait until text generation is complete before proceeding to the next frame
    text_generation_done_event.wait()

    # Allow further detection
    model_event.set()  # Signal that this frame is done processing

def connect_local_subscreen():
    server_thread = threading.Thread(target=app.run, kwargs={'threaded': True, 'port': 5000})
    server_thread.daemon = True  # Set as a daemon thread
    server_thread.start()

# Function to handle the text generation and notify completion
def generate_target_text_in_yolo(predictions):
    global text_generation_done_event
    text_generation_done_event.clear()
    connect_local_subscreen()
    # Simulate generating text (e.g., advertising text or any other processing)
    print("Generating advertisement text...")  # This should be your actual text generation logic
    # Simulate a delay in text generation
    pipeline.generate_advertisement(predictions)  # Call your actual LLM function here
    # After generation is complete, signal that text generation is done
    text_generation_done_event.set()

# Main function to start face detection
if __name__ == '__main__':
    detect_faces_from_webcam()