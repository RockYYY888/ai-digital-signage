import cv2
import dlib
import numpy as np
import time
import threading
import queue
from CV.yolov8 import analyze_frame  # 假设这是你的人脸检测函数
from ad_pool.random_ads import AdPool  # 假设这是你的广告池类

# 初始化人脸检测和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eyetracking/shape_predictor_68_face_landmarks.dat")

# 初始化摄像头
cap = cv2.VideoCapture(1)

# 创建窗口
cv2.namedWindow("Ad Player", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# 定义全局变量和锁
frame_lock = threading.Lock()
watching_status = False
watching_lock = threading.Lock()
total_watch_time = 0
cam_frame = None
cam_frame_lock = threading.Lock()
ad_frame = None

# 创建广告池实例
ad_pool = AdPool()

# 创建队列存储预测结果（只保留最新结果）
prediction_queue = queue.Queue(maxsize=1)

def calculate_eye_distance(landmarks):
    """计算双眼之间的距离"""
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))

def eye_tracking():
    """眼动追踪线程"""
    global watching_status, total_watch_time, cam_frame
    start_time = None
    last_prediction_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        current_time = time.time()
        is_watching = False
        for face in faces:
            landmarks = predictor(gray, face)
            eye_distance = calculate_eye_distance(landmarks)
            if eye_distance > 20:  # 判断是否在观看
                is_watching = True
                if start_time is None:
                    start_time = current_time
                # 每秒更新一次预测结果
                if current_time - last_prediction_time >= 1:
                    prediction = analyze_frame(frame,verbose=False)
                    if prediction:
                        # 去掉情绪部分，只保留前三个元素 (age, gender, race)
                        prediction = prediction[:3]
                        # 只保留最新预测结果
                        if prediction_queue.full():
                            prediction_queue.get()  # 移除旧结果
                        prediction_queue.put(prediction)
                    last_prediction_time = current_time
                break
        with watching_lock:
            watching_status = is_watching
            if is_watching and start_time is not None:
                total_watch_time += current_time - start_time
                start_time = current_time
            else:
                start_time = None
        with cam_frame_lock:
            cam_frame = frame.copy()
        time.sleep(0.01)  # 减少 CPU 占用

def play_advertisement():
    """播放广告线程"""
    global ad_frame
    while True:
        ad_path = ad_pool.get_random_ad()
        if not ad_path:
            print("No ad available, waiting...")
            time.sleep(1)
            continue
        ad_cap = cv2.VideoCapture(ad_path)
        if not ad_cap.isOpened():
            print(f"Cannot open the file: {ad_path}")
            continue
        fps = ad_cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / fps
        next_frame_time = time.time()

        # 记录广告开始时的观看时间
        with watching_lock:
            start_watch_time = total_watch_time

        while True:
            current_time = time.time()
            if current_time >= next_frame_time:
                ret, frame = ad_cap.read()
                if not ret:
                    print("End of advertisement video.")
                    break
                with frame_lock:
                    ad_frame = frame.copy()
                next_frame_time = current_time + frame_interval
            else:
                time.sleep(max(0, next_frame_time - current_time))
        ad_cap.release()

        # 计算广告观看时间
        with watching_lock:
            end_watch_time = total_watch_time
        ad_watch_time = end_watch_time - start_watch_time

        # 获取最后一次预测结果并合并
        if not prediction_queue.empty():
            final_prediction = prediction_queue.get()
            result = (round(ad_watch_time, 2), final_prediction)
            print(f"Result: {result}")  # 例如 (5.23, ('20-30', 'Male', 'White'))
        else:
            result = (round(ad_watch_time, 2), None)
            print(f"Result: {result}")  # 例如 (2.15, None)

# 启动线程
eye_thread = threading.Thread(target=eye_tracking, daemon=True)
ad_thread = threading.Thread(target=play_advertisement, daemon=True)
eye_thread.start()
ad_thread.start()

# 主循环：显示画面
while True:
    with cam_frame_lock:
        if cam_frame is not None:
            cv2.imshow("Camera", cam_frame)
    with frame_lock:
        if ad_frame is not None:
            cv2.imshow("Ad Player", ad_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()