import cv2
import dlib
import numpy as np
import time
import threading
import queue
import os
import re
from CV.yolov8 import analyze_frame
import sqlite3
from datetime import datetime

# 创建队列存储预测结果
prediction_queue = queue.Queue(maxsize=1)

# 全局变量追踪观看时间
watching_lock = threading.Lock()
total_watch_time = 0

def calculate_eye_distance(landmarks):
    
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))

def eye_tracking(active_event):
    """眼动追踪线程"""
    global total_watch_time
    
    print("开始眼动追踪")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("eyetracking/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(1)
    
    start_time = None
    last_prediction_time = 0
    initial_prediction = None
    
    with watching_lock:
        total_watch_time = 0

    while active_event.is_set():
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
                
                if initial_prediction is None or current_time - last_prediction_time >= 2:
                    prediction = analyze_frame(frame)
                    if prediction:
                        prediction = prediction[:3]
                        if initial_prediction is None:
                            initial_prediction = prediction
                            print(f"初始预测结果: {initial_prediction}")
                        if prediction_queue.full():
                            prediction_queue.get()
                        prediction_queue.put(prediction)
                        print(f"当前预测: {prediction}")
                    last_prediction_time = current_time
                break

        if is_watching and start_time is not None:
            with watching_lock:
                total_watch_time += current_time - start_time
            start_time = current_time
        else:
            start_time = None

    if prediction_queue.empty() and initial_prediction is not None:
        prediction_queue.put(initial_prediction)
        print(f"添加初始预测结果: {initial_prediction}")
    
    cap.release()
    print(f"眼动追踪结束，总观看时间: {total_watch_time:.2f}秒")

def extract_ad_id(video_path):
    """从视频路径中提取广告ID"""
    try:
        match = re.search(r'ad_(\d+)', video_path)
        if match:
            ad_id = int(match.group(1))
        else:
            ad_id = os.path.basename(video_path).split('.')[0]
        print(f"广告ID: {ad_id}")
    except Exception as e:
        print(f"提取广告ID出错: {e}")
        ad_id = hash(video_path) % 10000
        print(f"使用备用ID: {ad_id}")
    return ad_id

def start_eye_tracking(video_path):
    """启动眼动追踪"""
    print(f"开始追踪: {video_path}")
    ad_id = extract_ad_id(video_path)
    
    # 重置观看时间
    with watching_lock:
        global total_watch_time
        total_watch_time = 0
    
    # 清空预测队列
    while not prediction_queue.empty():
        prediction_queue.get()
    
    eyetrack_active = threading.Event()
    eyetrack_active.set()
    
    eyetrack_thread = threading.Thread(target=eye_tracking, args=(eyetrack_active,))
    eyetrack_thread.daemon = True
    eyetrack_thread.start()
    print("眼动追踪线程已启动")
    
    return eyetrack_active, eyetrack_thread, ad_id

def stop_eye_tracking(eyetrack_active, eyetrack_thread):
    """停止眼动追踪"""
    if eyetrack_active.is_set():  # 检查是否仍在运行
        eyetrack_active.clear()
        print("停止眼动追踪线程")
        eyetrack_thread.join(timeout=1.0)
        print("眼动追踪线程已终止")
    
    with watching_lock:
        watch_time = total_watch_time
    
    final_prediction = None
    try:
        if not prediction_queue.empty():
            final_prediction = prediction_queue.get(timeout=0.5)
            print(f"获取预测结果: {final_prediction}")
    except Exception as e:
        print(f"处理预测错误: {e}")
    
    return watch_time, final_prediction

def play_targeted_video(video_path):
    """播放目标视频"""
    print(f"play video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open: {video_path}")
        return
    
    cv2.namedWindow("Targeted Video", cv2.WINDOW_NORMAL)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Targeted Video", frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("end")

def track_and_play_video(video_path):
    """启动眼动追踪并播放视频"""
    print(f"开始追踪和播放: {video_path}")
    ad_id = extract_ad_id(video_path)
    
    eyetrack_active, eyetrack_thread, ad_id = start_eye_tracking(video_path)
    
    play_targeted_video(video_path)
    
    watch_time, final_prediction = stop_eye_tracking(eyetrack_active, eyetrack_thread)
    
    success = update_database(watch_time, final_prediction, ad_id)
    if success:
        print(f"time ={watch_time:.2f}, prediction={final_prediction}")
    else:
        print("update fail")
    
    return watch_time, final_prediction

def update_database(watch_time, demographics, ad_id):
    
    if demographics is None:
        print("Cannot define")
        return False
    
    try:
        # 解包 demographics 数据
        age_group, gender, ethnicity = demographics
        print(f"Prepare for database age={age_group}, gender={gender}, ethnicity={ethnicity}")
        
        # 使用与 dashboard 相同的数据库文件
        db_path = 'Dashboard/advertisements.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查 demographics 表是否存在，若不存在则创建
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demographics'")
        if not cursor.fetchone():
            print("demographics dont exist")
            cursor.execute("""
                CREATE TABLE demographics (
                    demographics_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gender TEXT,
                    age_group TEXT,
                    ethnicity TEXT
                )
            """)
        
        # 检查 viewers 表是否存在，若不存在则创建
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='viewers'")
        if not cursor.fetchone():
            print("viewers dont exist")
            cursor.execute("""
                CREATE TABLE viewers (
                    viewer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    view_time REAL,
                    demographics_id INTEGER,
                    visit_date TEXT,
                    ad_id INTEGER,  
                    FOREIGN KEY(demographics_id) REFERENCES demographics(demographics_id)
                )
            """)
        
        # 检查 demographics 表中是否已有匹配记录
        cursor.execute("""
            SELECT demographics_id FROM demographics 
            WHERE gender = ? AND age_group = ? AND ethnicity = ?
        """, (gender, age_group, ethnicity))
        
        result = cursor.fetchone()
        if result:
            demographics_id = result[0]
            print(f"find demographics  ID={demographics_id}")
        else:
            # 插入新的 demographics 记录
            cursor.execute("""
                INSERT INTO demographics (gender, age_group, ethnicity)
                VALUES (?, ?, ?)
            """, (gender, age_group, ethnicity))
            demographics_id = cursor.lastrowid
            print(f"create demographics ID={demographics_id}")
        
        # 插入 viewers 记录
        current_date = datetime.now().strftime('%Y-%m-%d')  # 格式与 dashboard 一致
        cursor.execute("""
            INSERT INTO viewers (view_time, demographics_id, visit_date, ad_id)
            VALUES (?, ?, ?, ?)
        """, (float(watch_time), demographics_id, current_date, int(ad_id)))  # 确保类型正确
        
        conn.commit()
        print(f"update time={watch_time:.2f}, demographic ID={demographics_id}, ad ID={ad_id}")
        return True
    
    except Exception as e:
        print(f"fail {e}")
        return False
    
    finally:
        if 'conn' in locals():
            conn.close()