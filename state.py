import threading
import time
import queue
from CV.yolov8 import cv_thread_func, analyze_frame
from LLM.LLM import AdvertisementPipeline
from data_integration.server import app_1
from data_integration.user_screen_server import app
from Dashboard.dashboard import app_2

class Context:
    def __init__(self):
        self.face_detection_active = threading.Event()
        self.face_detection_active.set()  # Default: face detection enabled
        self.detected_face_queue = queue.Queue(maxsize=1)  # Latest face only
        self.ad_text_queue = queue.Queue(maxsize=1)  # Latest ad text only
        self.current_ad_text = None
        self.state_lock = threading.Lock()  # Lock for state transitions
        self.is_first_ad_rotating = True  # Flag for first AdRotating entry
        self.eye_tracking_active = threading.Event()
        self.video_complete_event = threading.Event()
        self.video_completion_queue = queue.Queue()

class State:
    def __init__(self, context):
        self.context = context

    def handle(self):
        pass

class AdRotating(State):
    def handle(self):
        with self.context.state_lock:
            if self.context.is_first_ad_rotating:
                print("Ad Rotating: Displaying generic ad.")
                self.context.is_first_ad_rotating = False
            if not self.context.detected_face_queue.empty():
                frame, prediction = self.context.detected_face_queue.get()
                self.context.face_detection_active.clear()  # Pause face detection
                return ModelProcessing(self.context, frame, prediction)
            return self  # Stay in AdRotating

class ModelProcessing(State):
    def __init__(self, context, frame, prediction):
        super().__init__(context)
        self.frame = frame
        self.prediction = prediction
        self.ad_generated_event = threading.Event()

    def handle(self):
        with self.context.state_lock:
            print("Model Processing: Generating ad text.")
            processing_thread = threading.Thread(target=self.process_frame)
            processing_thread.start()
            processing_thread.join()  # Wait for thread to finish
            self.ad_generated_event.wait()  # Wait for ad text generation
            if not self.context.ad_text_queue.empty():
                return PersonalizedADDisplaying(self.context)
            else:
                print("Ad generation failed, returning to Ad Rotating.")
                self.context.face_detection_active.set()
                return AdRotating(self.context)

    def process_frame(self):
        ad_text = pipeline.generate_advertisement(self.prediction)

        if ad_text:
            try:
                self.context.ad_text_queue.put_nowait(ad_text)
            except queue.Full:
                pass  # Discard old data if queue is full
        self.ad_generated_event.set()

class PersonalizedADDisplaying(State):
    def handle(self):
        with self.context.state_lock:
            ad_text = self.context.ad_text_queue.get()
            self.context.current_ad_text = ad_text
            
            print("prepare to start")
            
            # 重置视频完成事件
            self.context.video_complete_event.clear()
            
            # 启动眼动追踪
            self.context.eye_tracking_active.set()
            
            # 启动监听视频完成的线程
            video_monitor = threading.Thread(
                target=self.monitor_video_completion
            )
            video_monitor.daemon = True
            video_monitor.start()
            
            # 等待视频播放完成
            self.context.video_complete_event.wait()
            
            # 视频播放完成，停止眼动追踪
            self.context.eye_tracking_active.clear()
            print("目标视频播放完成，眼动追踪已停止")
            
            # 重新启用人脸检测
            self.context.face_detection_active.set()
            
            return AdRotating(self.context)
    
    def monitor_video_completion(self):
        try:
            # 使用 Context 中定义的队列
            signal = self.context.video_completion_queue.get()
            if signal == "video_complete":
                self.context.video_complete_event.set()
        except Exception as e:
            print(f"error {e}")
            self.context.video_complete_event.set()

if __name__ == "__main__":
    context = Context()
    pipeline = AdvertisementPipeline()

    # Start Flask thread
    flask_thread1 = threading.Thread(
        target=app_1.run,
        kwargs={'threaded': True, 'port': 5000}
    )
    flask_thread1.daemon = True
    flask_thread1.start()

    # Start Flask thread
    flask_thread2 = threading.Thread(
        target=app.run,
        kwargs={'threaded': True, 'port': 5001}
    )
    flask_thread2.daemon = True
    flask_thread2.start()

    flask_thread3 = threading.Thread(
        target=app_2.run,
        kwargs={'threaded': True, 'port': 5002}
    )
    flask_thread3.daemon = True
    flask_thread3.start()


    # Start CV thread
    cv_thread = threading.Thread(
        target=cv_thread_func,
        args=(context.detected_face_queue, context.face_detection_active)
    )
    cv_thread.daemon = True
    cv_thread.start()

    # Run state machine
    current_state = AdRotating(context)
    while True:
        current_state = current_state.handle()
        time.sleep(0.15)  # Minimal delay for state machine pacing; adjust as needed