# eyetrack.py
import cv2
import dlib
import numpy as np
import time
import threading
import queue
import os
import re
import sqlite3
from datetime import datetime

# 预测结果队列
prediction_queue = queue.Queue(maxsize=1)

# 全局观看时间
watching_lock = threading.Lock()
total_watch_time = 0

def calculate_eye_distance(landmarks):
    """计算双眼之间的距离"""
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))

def eye_tracking(active_event):
    """眼动追踪线程"""
    global total_watch_time
    
    print("开始眼动追踪")
    
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor("eyetracking/shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"加载面部特征预测器出错: {e}")
        return
    
    # 初始化观看时间
    with watching_lock:
        total_watch_time = 0.0
        print("观看时间已重置为0秒")
    
    # 尝试打开摄像头
    cap = None
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries and (cap is None or not cap.isOpened()):
        print(f"眼动追踪：尝试打开摄像头 (尝试 {retry_count+1}/{max_retries})")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    print("眼动追踪：摄像头成功打开")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                print("眼动追踪：无法打开摄像头")
        except Exception as e:
            print(f"眼动追踪：摄像头错误: {e}")
        
        time.sleep(1.0)
        retry_count += 1
    
    if cap is None or not cap.isOpened():
        print("错误: 眼动追踪无法打开摄像头")
        return
    
    # 初始化变量
    start_time = None
    last_update_time = time.time()
    last_prediction_time = 0
    initial_prediction = None
    accumulated_watch_time = 0.0  # 本地累积观看时间
    
    # 获取已有的YOLO预测
    try:
        while not prediction_queue.empty():
            latest_prediction = prediction_queue.get()
            if latest_prediction and latest_prediction != ("no_face") and latest_prediction != ("analyzing"):
                initial_prediction = latest_prediction[:3]
                print(f"使用已有的YOLO预测: {initial_prediction}")
                prediction_queue.put(initial_prediction)  # 放回队列以备用
                break
    except Exception as e:
        print(f"获取预测失败: {e}")
    
    # 主循环
    try:
        frame_count = 0
        while active_event.is_set():
            frame_count += 1
            try:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    time.sleep(0.1)
                    continue
                    
                # 处理帧
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                current_time = time.time()
                is_watching = False
                
                # 分析面部特征
                for face in faces:
                    try:
                        landmarks = predictor(gray, face)
                        eye_distance = calculate_eye_distance(landmarks)
                        
                        # 判断是否在观看
                        if eye_distance > 8:
                            is_watching = True
                            if start_time is None:
                                start_time = current_time
                                print(f"开始观看，眼距: {eye_distance:.2f}")
                            
                            # 定期更新预测 - 降低YOLO调用频率
                            if current_time - last_prediction_time >= 5 and frame_count % 30 == 0:
                                last_prediction_time = current_time
                                # 不在这里调用YOLO分析以避免资源冲突
                        break
                    except Exception as e:
                        print(f"面部分析错误: {e}")
                        continue
    
                # 更新观看时间 - 关键修复
                if is_watching and start_time is not None:
                    time_diff = current_time - start_time
                    start_time = current_time
                    accumulated_watch_time += time_diff
                    
                    if current_time - last_update_time >= 5.0:
                        with watching_lock:
                            total_watch_time = accumulated_watch_time
                        last_update_time = current_time
                        print(f"累计观看时间: {accumulated_watch_time:.2f}秒")
                elif not is_watching:
                    start_time = None
                
                # 控制循环速度
                if frame_count % 2 == 0:
                    time.sleep(0.03)
                
            except Exception as e:
                print(f"眼动循环错误: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"眼动追踪致命错误: {e}")
    
    finally:
        # 释放资源
        if cap is not None:
            cap.release()
        
        # 最终更新全局观看时间
        with watching_lock:
            total_watch_time = accumulated_watch_time
        
        # 确保预测结果可用
        if initial_prediction:
            try:
                if prediction_queue.empty():
                    prediction_queue.put(initial_prediction)
            except:
                pass
        
        print("眼动追踪线程已完成")
        
        # 直接更新数据库
        if initial_prediction:
            ad_id = extract_ad_id("data_integration/static/videos")  # 替换为实际视频路径
            success = update_database(total_watch_time, initial_prediction, ad_id)
            if success:
                print(f"数据库更新成功: watch_time={total_watch_time:.2f}, demographics={initial_prediction}, ad_id={ad_id}")
            else:
                print("数据库更新失败")

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

def update_database(watch_time, demographics, ad_id):
    """更新数据库，将观看时间和用户特征写入"""
    if demographics is None:
        print("无法识别用户特征")
        return False
    
    try:
        age_group, gender, ethnicity = demographics
        print(f"准备更新数据库: age={age_group}, gender={gender}, ethnicity={ethnicity}")
        
        db_path = 'Dashboard/advertisements.db'  # 与 dashboard 一致的数据库路径
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查表是否存在，若不存在则创建 demographics 表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demographics'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE demographics (
                    demographics_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gender TEXT,
                    age_group TEXT,
                    ethnicity TEXT
                )
            """)
        
        # 检查表是否存在，若不存在则创建 viewers 表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='viewers'")
        if not cursor.fetchone():
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
        
        # 检查现有 demographics 记录
        cursor.execute("""
            SELECT demographics_id FROM demographics 
            WHERE gender = ? AND age_group = ? AND ethnicity = ?
        """, (gender, age_group, ethnicity))
        
        result = cursor.fetchone()
        if result:
            demographics_id = result[0]
            print(f"找到已有 demographics ID={demographics_id}")
        else:
            cursor.execute("""
                INSERT INTO demographics (gender, age_group, ethnicity)
                VALUES (?, ?, ?)
            """, (gender, age_group, ethnicity))
            demographics_id = cursor.lastrowid
            print(f"创建新 demographics ID={demographics_id}")
        
        # 插入 viewers 记录
        current_date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            INSERT INTO viewers (view_time, demographics_id, visit_date, ad_id)
            VALUES (?, ?, ?, ?)
        """, (float(watch_time), demographics_id, current_date, int(ad_id)))
        
        conn.commit()
        print(f"数据库更新: time={watch_time:.2f}, demographics_id={demographics_id}, ad_id={ad_id}")
        return True
    
    except Exception as e:
        print(f"数据库更新失败: {e}")
        return False
    
    finally:
        if 'conn' in locals():
            conn.close()