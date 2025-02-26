import cv2
import dlib
import numpy as np
import time
import threading
import random


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


ad_list = ["ad1.mp4", "ad2.mp4", "ad3.mp4"]
current_ad_index = 0
cap = cv2.VideoCapture(1)


cv2.namedWindow("Ad Player", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)


frame_lock = threading.Lock()

watching_status = False
watching_lock = threading.Lock()
total_watch_time = 0
cam_frame = None
cam_frame_lock = threading.Lock()
ad_frame = None


class AdPool:
    def __init__(self):
        self.ads = set()  # use set to save the the ads path
        self.current_ad = None
        self.lock = threading.Lock()
    
    def add_ad(self, ad_path):
        with self.lock:
            self.ads.add(ad_path)
    
    def remove_ad(self, ad_path):
        with self.lock:
            self.ads.discard(ad_path)
    
    def get_random_ad(self):
        with self.lock:
            if not self.ads:
                return None
            # make sure it won't play the same ad twice
            available_ads = list(self.ads - {self.current_ad} if self.current_ad else self.ads)
            if not available_ads and self.current_ad:
                available_ads = list(self.ads)
            if available_ads:
                self.current_ad = random.choice(available_ads)
            
                global total_watch_time, start_time
                with watching_lock:
                    total_watch_time = 0
                    start_time = None
                return self.current_ad
            return None


ad_pool = AdPool()

ad_pool.add_ad("ad1.mp4")
ad_pool.add_ad("ad2.mp4")
ad_pool.add_ad("ad3.mp4")

def calculate_eye_distance(landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    return np.linalg.norm(np.array(right_eye) - np.array(left_eye))

def eye_tracking():
    global watching_status, total_watch_time, cam_frame
    start_time = None
    
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
            
            if eye_distance > 20:
                is_watching = True
                if start_time is None:
                    start_time = current_time
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

def play_advertisement():
    global ad_frame
    while True:
        ad_path = ad_pool.get_random_ad()
        if not ad_path:
            time.sleep(1)
            continue
            
        ad_cap = cv2.VideoCapture(ad_path)
        if not ad_cap.isOpened():
            print(f"cannot open the file: {ad_path}")
            ad_pool.remove_ad(ad_path)
            continue
        
        fps = ad_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        frame_interval = 1.0 / fps
        next_frame_time = time.time()
        
        while True:
            current_time = time.time()
            
            if current_time >= next_frame_time:
                ret, frame = ad_cap.read()
                if not ret:
                    break
                    
                with frame_lock:
                    ad_frame = frame.copy()
                
                next_frame_time = current_time + frame_interval
            else:
                time.sleep(max(0, next_frame_time - current_time))
        
        ad_cap.release()

eye_thread = threading.Thread(target=eye_tracking, daemon=True)
ad_thread = threading.Thread(target=play_advertisement, daemon=True)

eye_thread.start()
ad_thread.start()

# main loop process the screen 
while True:
    # show the camera video
    with cam_frame_lock:
        if cam_frame is not None:
            cv2.imshow("Camera", cam_frame)
    
    # display the ads.
    with frame_lock:
        if ad_frame is not None:
            current_ad = ad_frame.copy()
            cv2.putText(current_ad, f"Watched: {int(total_watch_time)}s", 
                      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Ad Player", current_ad)
    
    # check the quit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        
        pass

# close and quit
cap.release()
cv2.destroyAllWindows()