import threading
from datetime import datetime
import cv2
import threading
from threading import Event
import time
from ultralytics import YOLO
from model import *
from PIL import Image
import os
import data_store
# 初始化全局变量
cap = cv2.VideoCapture(1)
face_detector = YOLO("yolov8l-face.pt", verbose=False)
desired_size = (224, 224)
detection_done_event = Event()
llm_done_event = Event()
detection_done_event.set() 
llm_done_event.clear()  

# 全局变量控制检测的时间间隔
last_detection_time = time.time()
detection_interval = 5  # 每5秒检测一次
model_emotion = EmotionClassifier(num_classes=4)  # 根据实际的情绪分类数量调整
model_emotion = model_emotion.to(device) 
model_demographic = FaceAttributeModel(num_age_classes, num_gender_classes, num_race_classes)
model_demographic = model_demographic.to(device)
checkpoint_path = 'best_face_attribute_model.pth'
checkpoint_path2 = 'best_project_model.pth' 
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
checkpoint2 = torch.load(checkpoint_path2, map_location=torch.device('cpu'))
model_demographic.load_state_dict(checkpoint['model_state_dict'])
model_emotion.load_state_dict(checkpoint2['model_state_dict'])


def detect_faces_from_webcam():
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

            cv2.imshow("Webcam - Press ESC to Exit", frame)

            # 等待检测完成信号
            detection_done_event.wait()

            # 开始检测
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                print("[YOLO] Waiting for LLM to process the last result...")
                last_detection_time = current_time
                llm_done_event.wait()  
                llm_done_event.clear() 

                threading.Thread(target=process_frame, args=(frame.copy(),)).start()


            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def llm_processing():
    global llm_done_event

    while True:
        # 等待检测完成信号
        llm_done_event.wait()

        # 模拟 LLM 处理
        combined_prediction = data_store.combined_prediction
        print(f"LLM is processing: {combined_prediction}")
        time.sleep(2)  # 模拟处理时间

        # LLM 处理完成后清除信号
        llm_done_event.clear()

def process_frame(initial_frame):
    # 不要在这里使用 cap，避免线程间冲突
    global detection_done_event, llm_done_event
    detection_done_event.clear()
    results = face_detector(initial_frame, conf=0.86)
    if len(results) == 0:
        print("No face detected.")
        detection_done_event.set()
        return

    for result in results:
        if len(result.boxes.xyxy) == 0:
            print("[Info] No face detected at this timestamp.")
            detection_done_event.set()
            continue
        print("[Info] Face detected and processing.")
        
        padding = 20  # 额外多扣的像素数量，根据需求调整
        box = result.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        # 扩展边界，添加 padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(initial_frame.shape[1], x2 + padding)
        y2 = min(initial_frame.shape[0], y2 + padding)
        cropped_image = initial_frame[y1:y2, x1:x2]

        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        age_label, gender_label, race_label = predict(model_demographic, pil_image)
        emotion_pred = predict2(model_emotion, pil_image)
        emotion_label = emotion_mapping.get(emotion_pred, "Unkonwn")
        combined_prediction = (age_label, gender_label, emotion_label)

        data_store.combined_prediction = combined_prediction
        llm_done_event.set()  # 让 LLM 知道可以处理了

        # 等待 LLM 处理完成
        llm_done_event.wait()

        # 恢复检测信号
        detection_done_event.set()

        # 确保 faces 文件夹存在
        os.makedirs("faces", exist_ok=True)

        # 仅进行一次检测和预测
        ret, new_frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            return

        new_results = face_detector(new_frame, conf=0.86, verbose=False)
        if len(new_results) == 0:
            return  # No face detected in the new frame

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
            new_pil_image = Image.fromarray(cv2.cvtColor(new_cropped_image, cv2.COLOR_BGR2RGB))

            # 调用预测函数并记录结果
            prediction = predict(model_demographic, new_pil_image)

            # 调用情绪预测函数
            prediction2 = predict2(model_emotion, new_pil_image)
            emotion_name = emotion_mapping.get(prediction2, "Unknown") 
            combined_prediction = (*prediction, emotion_name)
            data_store.combined_prediction = combined_prediction
            print(f"Predicted Demographics: {combined_prediction}")
            break  # 只处理一张人脸

        return
    
      # 处理完当前人脸后返回，等待下一次检测

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

        # 只进行一次情绪检测

if __name__ == '__main__':
    threading.Thread(target=llm_processing, daemon=True).start()
    detect_faces_from_webcam()


