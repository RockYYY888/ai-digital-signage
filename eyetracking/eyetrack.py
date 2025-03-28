import re
import cv2
import dlib
import numpy as np
import time
import threading
import sqlite3
from datetime import datetime
from util import get_resource_path

watching_lock = threading.Lock()
total_watch_time = 0

def extract_number(filename):
    """Extract the numeric part from the filename."""
    match = re.search(r"(\d+)\.mp4$", filename)  # Match the numeric part in the filename
    if match:
        return str(int(match.group(1)))  # Convert to integer to remove leading zeros, then back to string
    return None

def calculate_eye_distance(landmarks):
    """Calculate the distance between the eyes."""
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))

def eye_tracking_thread_func(cap, eye_tracking_active, context):
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor("eyetracking/shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"Error loading facial feature predictor: {e}")
        return

    with watching_lock:
        context.total_watch_time = 0.0

    if not cap or not cap.isOpened():
        print("[CV] Failed to open webcam.")
        return

    start_time = None

    try:
        while True:
            if not eye_tracking_active.is_set():
                with watching_lock:
                    start_time = None
                time.sleep(0.2)
                continue

            if not context.user_screen_focus.is_set():
                with watching_lock:
                    start_time = None
                time.sleep(0.2)
                continue

            ret, frame = cap.read()
            if not ret:
                print("[Eyetracking] Failed to capture frame.")
                exit(1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            current_time = time.time()
            is_watching = False

            for face in faces:
                try:
                    landmarks = predictor(gray, face)
                    eye_distance = calculate_eye_distance(landmarks)
                    if eye_distance > 8:  # 可根据需要调整阈值
                        is_watching = True
                        break
                except Exception as e:
                    print(f"Facial analysis errors: {e}")
                    continue

            with watching_lock:
                if is_watching and context.user_screen_focus.is_set():
                    if start_time is None:
                        start_time = current_time
                    else:
                        context.total_watch_time += (current_time - start_time)
                        start_time = current_time
                else:
                    start_time = None

            time.sleep(0.03)

    except Exception as e:
        print("[Eyetracking] Fatal error:", e)
        exit(1)

    finally:
        if cap is not None:
            cap.release()


def update_database(watch_time, prediction, ad_id):
    try:
        # 1. Parsing the incoming prediction: (age_group, gender, ethnicity)
        age_group, gender, ethnicity = prediction

        db_path = get_resource_path('../advertisements.db')  # Consistent database path with dashboard
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. Query the corresponding demographics_id
        select_sql = """
            SELECT demographics_id
            FROM demographics
            WHERE age_group = ?
              AND gender = ?
              AND ethnicity = ?
            LIMIT 1
        """
        cursor.execute(select_sql, (age_group, gender, ethnicity))
        result = cursor.fetchone()

        if not result:
            print(f"No corresponding id found in demographics table, prediction={prediction}")
            print("[Eyetracking] Failed updating demographics in db")
            exit(1)
            return False
        demographics_id = result[0]

        # 3. Get the current date in the format of YYYY-MM-DD
        current_date = datetime.now().strftime('%Y-%m-%d')

        # 4. Insert viewing time and other information into the viewers table
        insert_sql = """
            INSERT INTO viewers (demographics_id, ad_id, view_time, visit_date)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_sql, (demographics_id, ad_id, round(watch_time, 2), current_date))
        conn.commit()

        print(f"Database update successful: time={watch_time:.2f}, demographics_id={demographics_id}, ad_id={ad_id}, date={current_date}")
        return True

    except Exception as e:
        print(f"Database update failed: {e}")
        return False

    finally:
        if 'conn' in locals():
            conn.close()