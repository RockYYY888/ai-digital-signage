import threading
from datetime import datetime
import cv2
import time
from ultralytics import YOLO
from model import *  # 确保这里包含了情绪模型和更新后的 predict 函数
from PIL import Image
import os
import torch
from map import *
# 初始化全局变量
cap = cv2.VideoCapture(1)
face_detector = YOLO("yolov8l-face.pt", verbose=False)
desired_size = (224, 224)

# 控制检测时间间隔
last_detection_time = time.time()
detection_interval = 2  # 2

# 加载情绪识别模型
model_emotion = EmotionClassifier(num_classes=4)  # 根据实际的情绪分类数量调整
model_emotion = model_emotion.to('cpu')  # 使用 CPU
checkpoint_path = 'best_project_model.pth'  # 替换为实际的模型路径
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_emotion.load_state_dict(checkpoint['model_state_dict'])

def detect_emotions_from_webcam():
    global last_detection_time

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    print("Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 显示摄像头实时画面
            cv2.imshow("Webcam - Press ESC to Exit", frame)

            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                last_detection_time = current_time

                # 在新线程中进行人脸检测，避免阻塞实时显示
                threading.Thread(target=process_frame_for_emotion, args=(frame.copy(),)).start()

            if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_frame_for_emotion(initial_frame):
    results = face_detector(initial_frame, conf=0.86)
    if len(results) == 0:
        print("No face detected.")
        return

    for result in results:
        if len(result.boxes.xyxy) == 0:
            print("[Info] No face detected at this timestamp.")
            continue
        print("[Info] Face detected and processing.")
        padding = 20
        box = result.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(initial_frame.shape[1], x2 + padding)
        y2 = min(initial_frame.shape[0], y2 + padding)
        cropped_image = initial_frame[y1:y2, x1:x2]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # 多次检测情绪
        end_time = time.time() + 1  # 在接下来的两秒内进行多次检测
        predictions = []

        while time.time() < end_time:
            ret, new_frame = cap.read()
            if not ret:
                continue

            new_results = face_detector(new_frame, conf=0.86, verbose=False)
            if len(new_results) == 0:
                continue

            for new_result in new_results:
                if len(new_result.boxes.xyxy) == 0:
                    continue

                new_box = new_result.boxes.xyxy[0]
                x1, y1, x2, y2 = map(int, new_box)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(new_frame.shape[1], x2 + padding)
                y2 = min(new_frame.shape[0], y2 + padding)
                new_cropped_image = new_frame[y1:y2, x1:x2]

                # 转为灰度图
                #gray_face = cv2.cvtColor(new_cropped_image, cv2.COLOR_BGR2GRAY)
                #resized_face = cv2.resize(gray_face, desired_size)
                #pil_image = Image.fromarray(resized_face)

                # 调用情绪预测函数
                prediction = predict(model_emotion, pil_image)
                predictions.append(prediction)
                break  # 只处理一张人脸

            time.sleep(0.2)

        # 统计并选择最有可能的情绪结果
        if predictions:
            # 使用字典映射得到对应的情绪名称
            most_likely_prediction = max(set(predictions), key=predictions.count)

            # 获取情绪名称
            emotion_name = list(emotion_mapping.keys())[list(emotion_mapping.values()).index(most_likely_prediction)]
            print(f"[Info] Most likely result: {emotion_name}")
        else:
            print("[User] Please stay longer")

        return  # 处理完当前人脸后返回

if __name__ == '__main__':
    detect_emotions_from_webcam()

