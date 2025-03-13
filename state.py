import threading
import time
import queue
import os

from flask import Flask

from CV.yolov8 import cv_thread_func, analyze_frame
from LLM.LLM import AdvertisementPipeline
from data_integration.server import secondary_screen_app
from data_integration.user_screen_server import user_screen
from Dashboard.dashboard import init_dashboard
#from eyetrack import *
from data_integration.data_interface import video_queue, ad_queue

class Context:
    def __init__(self):
        self.face_detection_active = threading.Event()
        self.face_detection_active.set()  # 默认启用人脸检测
        self.detected_face_queue = queue.Queue(maxsize=1)  # 仅存储最新人脸
        self.ad_text_queue = queue.Queue(maxsize=1)  # 仅存储最新广告文本
        self.current_ad_text = None
        self.state_lock = threading.Lock()  # 状态转换锁
        self.eye_tracking_active = threading.Event()
        self.video_completed = threading.Event()  # 用于接收视频播放完成信号

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
                print("[DEBUG] Into here")
                frame, prediction = self.context.detected_face_queue.get()
                self.context.face_detection_active.clear()  # 暂停人脸检测
                print("[ERROR] 断掉了cv")

                print("[State] LLM Processing: Generating ad text.")
                processing_thread = threading.Thread(target=self.process_frame, args=prediction)
                processing_thread.start()
                processing_thread.join()  # 等待线程完成
                self.llm_text_generated_event.wait()  # 等待广告文本生成
                if not self.context.ad_text_queue.empty():
                    return PersonalizedADDisplaying(self.context)
                else:
                    print("[Error] Ad generation failed, returning to Ad Rotating.")
                    self.context.face_detection_active.set()
            else:
                # print("[AD R] No valid face, return self")
                # print(f"[DEBUG] face_detection_active state: {self.context.face_detection_active.is_set()}")


                return self  # 保持在 AdRotating 状态

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

    def handle(self):
        with self.context.state_lock:
            ad_text = self.context.ad_text_queue.get()
            
            self.context.current_ad_text = ad_text
            # 在user server里set过
            self.context.video_completed.wait()  # 阻塞直到 video_completed 被设置
            self.context.video_completed.clear()  # 重置事件状态

            """print(f"开始个性化视频播放: {ad_text}")

            # 抽取广告ID
            ad_id = extract_ad_id(ad_text)

            # 更安全的摄像头转换过程
            print("正在安全停止人脸检测...")
            self.context.face_detection_active.clear()
            time.sleep(3.0)  # 增加等待时间确保摄像头完全释放
            print("摄像头资源已释放")

            try:
                # 启动眼动追踪线程
                self.context.eye_tracking_active.set()
                eyetrack_thread = threading.Thread(
                    target=eye_tracking,
                    args=(self.context.eye_tracking_active,)
                )
                eyetrack_thread.daemon = True
                eyetrack_thread.start()
                print("眼动追踪线程已启动")

                # 放视频到队列
                try:
                    while not video_queue.empty():
                        video_queue.get()
                    while not ad_queue.empty():
                        ad_queue.get()

                    video_queue.put(ad_text, block=False)
                    ad_queue.put(ad_text, block=False)
                    print(f"已将视频 {ad_text} 加入播放队列")
                except queue.Full:
                    print("队列已满，视频可能无法播放")

                # 等待视频播放完成，设置更保守的超时
                print("等待视频播放完成...")
                video_wait_time = 20  # 秒
                for _ in range(video_wait_time):
                    time.sleep(1)
                    # 检查视频状态
                    # (这里可以添加视频播放完成的检查逻辑)

                # 安全停止眼动追踪
                print("安全停止眼动追踪...")
                self.context.eye_tracking_active.clear()
                # 给足够时间让眼动追踪线程完成最后的时间更新
                time.sleep(1.0)  # 确保最后的时间更新完成
                eyetrack_thread.join(timeout=3.0)

                # 获取预测结果
                final_prediction = None
                try:
                    if not prediction_queue.empty():
                        final_prediction = prediction_queue.get(timeout=0.5)
                        print(f"获取预测结果: {final_prediction}")
                except Exception as e:
                    print(f"获取预测结果失败: {e}")

                # 更新数据库逻辑将移到eyetrack.py中

            except Exception as e:
                print(f"处理个性化广告时发生错误: {e}")

            finally:
                # 无论如何，确保重新启用人脸检测
                print("重新启用人脸检测...")"""
            return AdRotating(self.context, True)

# 创建 Flask 主应用
main_app = Flask(__name__)

# 挂载 app_1 到 /app1/
main_app.register_blueprint(secondary_screen_app, url_prefix='/secondary-screen')

# 挂载 app_user 到 /user/
main_app.register_blueprint(user_screen, url_prefix='/user-screen')

dash_app = init_dashboard(main_app)

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

    print("[Flask] User screen running on http://127.0.0.1:5000/user-screen/")
    print("[Flask] Secondary screen running on http://127.0.0.1:5000/secondary-screen/")
    print("[Flask] Dashboard running on http://127.0.0.1:5000/dashboard/")

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
            print("Current state: " + str(current_state) + " Prev state: " + str(prev_state))
            print("[State] Current state is: " + str(current_state))
            print("[Main] CV thread alive: " + str(cv_thread.is_alive()))
        time.sleep(0.5)  # 状态机节奏控制
