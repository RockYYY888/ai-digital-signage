# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

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
"""threading.Lock: Global lock to synchronize access to total watch time across threads."""

def extract_number(filename):
    """Extract the numeric part from a video filename.

    This function parses a filename to extract the numeric portion before the '.mp4' extension,
    removing any leading zeros.

    Args:
        filename (str): The filename to parse (e.g., '001.mp4').

    Returns:
        str: The extracted number as a string, or None if no match is found.
    """
    match = re.search(r"(\d+)\.mp4$", filename)  # Match the numeric part in the filename
    if match:
        return str(int(match.group(1)))  # Convert to integer to remove leading zeros, then back to string
    return None


def calculate_eye_distance(landmarks):
    """Calculate the Euclidean distance between the eyes based on facial landmarks.

    Args:
        landmarks (dlib.full_object_detection): The detected facial landmarks.

    Returns:
        float: The distance between the left and right eye landmarks.
    """
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))


def eye_tracking_thread_func(cap, eye_tracking_active, context):
    """Run eye tracking in a separate thread to monitor user attention.

    This function uses facial landmark detection to determine if a user is watching the screen.
    It updates the total watch time in the provided context when the user is actively viewing.

    Args:
        cap (cv2.VideoCapture): The video capture object for the webcam.
        eye_tracking_active (threading.Event): Event to control when eye tracking is active.
        context (Context): The shared context object storing total watch time and state.

    Raises:
        SystemExit: If the webcam fails to capture frames or a fatal error occurs.
    """
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(get_resource_path("eyetracking/shape_predictor_68_face_landmarks.dat"))
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
                    if start_time is not None:
                        context.total_watch_time += (time.time() - start_time)
                        start_time = None
                time.sleep(0.2)
                continue

            ret, frame = cap.read()
            if not ret:
                print("[Eyetracking] Failed to capture frame.")
                exit(1)

            current_time = time.time()

            # Check screen focus
            if not context.user_screen_focus.is_set():
                with watching_lock:
                    if start_time is not None:
                        context.total_watch_time += (current_time - start_time)
                        start_time = None
                time.sleep(0.2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            is_watching = False

            for face in faces:
                try:
                    landmarks = predictor(gray, face)
                    eye_distance = calculate_eye_distance(landmarks)
                    if eye_distance > 8:
                        is_watching = True
                        break
                except Exception as e:
                    print(f"Facial analysis errors: {e}")
                    continue

            with watching_lock:
                if is_watching:
                    if start_time is None:
                        start_time = current_time
                    else:
                        context.total_watch_time += (current_time - start_time)
                        start_time = current_time
                else:
                    if start_time is not None:
                        context.total_watch_time += (current_time - start_time)
                        start_time = None

            time.sleep(0.03)

    except Exception as e:
        print("[Eyetracking] Fatal error:", e)
        exit(1)

    finally:
        if cap is not None:
            cap.release()


def update_database(watch_time, prediction, ad_id):
    """Update the database with viewing statistics for an advertisement.

    This function parses the demographic prediction, retrieves the corresponding demographics ID,
    and inserts the watch time, ad ID, and visit date into the viewers table.

    Args:
        watch_time (float): The total time the user watched the ad in seconds.
        prediction (tuple): A tuple of (age_group, gender, ethnicity) from the face detection.
        ad_id (str): The ID of the advertisement.

    Returns:
        bool: True if the database update was successful, False otherwise.
    """
    try:
        # 1. Parsing the incoming prediction: (age_group, gender, ethnicity)
        age_group, gender, ethnicity = prediction

        db_path = get_resource_path('advertisements.db')  # Consistent database path with dashboard
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