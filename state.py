import threading
import time
import queue
import webbrowser

from flask import Flask
from CV.yolov8 import cv_thread_func, analyze_frame
from LLM.LLM import AdvertisementPipeline
from data_integration.server import secondary_screen_app
from data_integration.user_screen_server import user_screen
from Dashboard.dashboard import init_dashboard
from data_integration.data_interface import video_queue, ad_queue
#from eyetrack import *

class Context:
    def __init__(self):
        self.face_detection_active = threading.Event()
        self.face_detection_active.set()  # 默认启用人脸检测
        self.detected_face_queue = queue.Queue(maxsize=1)  # 仅存储最新人脸
        self.ad_text_queue = queue.Queue(maxsize=1)  # 仅存储最新广告文本
        self.current_ad_text = None
        self.state_lock = threading.Lock()  # 状态转换锁
        self.eye_tracking_active = threading.Event()
        self.default_video_completed = threading.Event()  # 新增：默认广告播放完成信号
        self.personalized_video_completed = threading.Event()  # 修改：个性化广告播放完成信号（原 video_completed）

class State:
    def __init__(self, context, is_first=False):
        self.context = context
        self.is_first = is_first

    def handle(self):
        pass

class AdRotating(State):
    def __init__(self, context, is_first=False):
        super().__init__(context)
        self.is_first = is_first
        self.llm_text_generated_event = threading.Event()

    def __str__(self):
        return "ADRotation"

    def handle(self):
        with self.context.state_lock:
            self.context.face_detection_active.set()  # 进入AD Rotation开启yolo线程
            # print("[AD R] Set face_detection_active true")
            if self.is_first:
                print("[State] Ad Rotating: Displaying generic ad.")
                self.is_first = False

            if not self.context.detected_face_queue.empty():
                frame, prediction = self.context.detected_face_queue.get()
                self.context.face_detection_active.clear()  # 暂停人脸检测
            

                print("[State] LLM Processing: Generating ad text.")
                processing_thread = threading.Thread(target=self.process_frame, args=(prediction,))
                processing_thread.start()
                processing_thread.join()  # 等待线程完成
                self.llm_text_generated_event.wait()  # 等待广告文本生成
                # 只有在跑完这个视频后才查询是否切换到下一个状态
                self.context.default_video_completed.wait()
                self.context.default_video_completed.clear()
                if not self.context.ad_text_queue.empty():

                    return PersonalizedADDisplaying(self.context)
                else:
                    print("[Error] Ad generation failed, returning to Ad Rotating.")
                    self.context.face_detection_active.set()
            else:
                # print("[AD R] No valid face, return self")
                # print(f"[DEBUG] face_detection_active state: {self.context.face_detection_active.is_set()}")
                return self

    def process_frame(self, prediction):
        ad_text = pipeline.generate_advertisement(prediction)
        if ad_text:
            try:
                self.context.ad_text_queue.put_nowait(ad_text)
            except queue.Full:
                print("[ERROR] Queue is full")
                pass  # 如果队列满，丢弃旧数据
        self.llm_text_generated_event.set()

class PersonalizedADDisplaying(State):
    def __init__(self, context):
        super().__init__(context)

    def __str__(self):
        return "PersonalizedADDisplaying"

    def handle(self):
        with self.context.state_lock:
            ad_text = self.context.ad_text_queue.get()
            
            self.context.current_ad_text = ad_text

            self.context.personalized_video_completed.wait()  # 等待个性化广告播放完成
            self.context.personalized_video_completed.clear()  # 重置信号

        return AdRotating(self.context, True)

# 创建 Flask 主应用
main_app = Flask(__name__)

main_app.register_blueprint(secondary_screen_app, url_prefix='/secondary-screen')
main_app.register_blueprint(user_screen, url_prefix='/user-screen')
dash_app = init_dashboard(main_app)

@main_app.route('/')
def index():
    return """
    <h2>Welcome to the Flask Server!</h2>
    <p>Available Endpoints:</p>
    <ul>
        <li><a href="/user-screen/">User Screen</a></li>
        <li><a href="/secondary-screen/">Secondary Screen</a></li>
        <li><a href="/dashboard/">Dashboard</a></li>
    </ul>
    """

if __name__ == "__main__":
    context = Context()
    pipeline = AdvertisementPipeline()

    from data_integration.user_screen_server import set_context
    set_context(context)

    # 运行Flask app
    flask_thread = threading.Thread(target=main_app.run, kwargs={
        "host": "127.0.0.1",
        "port": 5000,
        "threaded": True,
        "debug": False
    })
    flask_thread.daemon = True
    flask_thread.start()

    webbrowser.open("http://127.0.0.1:5000/user-screen/")
    webbrowser.open("http://127.0.0.1:5000/secondary-screen/")
    webbrowser.open("http://127.0.0.1:5000/dashboard/")

    # 启动 CV 线程
    cv_thread = threading.Thread(
        target=cv_thread_func,
        args=(context.detected_face_queue, context.face_detection_active)
    )
    cv_thread.daemon = True
    cv_thread.start()

    # 运行状态机
    current_state = AdRotating(context, True)
    prev_state = str(current_state)
    while True:
        prev_state = str(current_state)
        current_state = current_state.handle()
        if (prev_state != str(current_state)):
            print("Current state: " + str(current_state) + ", Prev state: " + str(prev_state))
            print("[Main] CV thread alive: " + str(cv_thread.is_alive()))
        time.sleep(0.5)  # 状态机节奏控制
