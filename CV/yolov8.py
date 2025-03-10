import threading
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
from data_integration.data_interface import prediction_queue, frame_queue

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model_emotion = EmotionClassifier(num_classes=4).to(device)
model_demographic = FaceAttributeModel(num_age_classes=4, num_gender_classes=2, num_race_classes=5).to(device)
checkpoint_path = 'CV/best_face_attribute_model.pth'
checkpoint_path2 = 'CV/best_project_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint2 = torch.load(checkpoint_path2, map_location=device)
model_demographic.load_state_dict(checkpoint['model_state_dict'])
model_emotion.load_state_dict(checkpoint2['model_state_dict'])

# 初始化 YOLO 人脸检测器
face_detector = YOLO("CV/yolov8l-face.pt", verbose=False)

def analyze_frame(frame):
    """分析单帧以检测人脸并生成预测结果"""
    # 人脸检测
    results = face_detector(frame, conf=0.86)
    if not results or len(results[0].boxes.xyxy) == 0:
        prediction_queue.put(("no_face"))
        print("[INFO] No face detected at this frame.")
        return None, None

    prediction_queue.put(("analyzing"))

    # 处理检测到的第一个人脸
    box = results[0].boxes.xyxy[0]
    padding = 35
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    cropped_image = frame[y1:y2, x1:x2]

    # 转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # 预测 demographics
    age_label, gender_label, race_label = predict_demographics(model_demographic, pil_image)

    # 预测情绪
    emotion_pred = predict2(model_emotion, pil_image)
    emotion_label = emotion_mapping.get(emotion_pred, "Unknown")
    combined_prediction = (age_label, gender_label, race_label, emotion_label)
    print(f"[INFO] Predicted Demographics: {combined_prediction}")

    prediction_queue.put(combined_prediction)
    return combined_prediction, pil_image

def predict_demographics(model, image):
    """预测年龄、性别和种族"""
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

def cv_thread_func(detected_face_queue, face_detection_active):
    """从摄像头捕获帧并检测人脸"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detection_interval = 2  # 每 2 秒检测一次
    last_detection_time = time.time()

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    try:
        while True:
            if not face_detection_active.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                last_detection_time = current_time
                prediction, cropped_image = analyze_frame(frame)
                if prediction:  # 仅在检测到人脸时放入队列
                    try:
                        cropped_image_bgr = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                        frame_queue.put_nowait(cropped_image_bgr)  # 单次帧放入队列
                        detected_face_queue.put_nowait((cropped_image_bgr, prediction))
                    except queue.Full:
                        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print(cv2.getBuildInformation())