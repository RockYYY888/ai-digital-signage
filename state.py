import os
import threading
import time
import queue
import webbrowser
import cv2
from flask import Flask
from CV.yolov8 import cv_thread_func
from LLM.LLM import AdvertisementPipeline
from data_integration.server import secondary_screen_app
from data_integration.user_screen_server import user_screen
from Dashboard.dashboard import init_dashboard
from data_integration.data_interface import secondary_screen_signal_queue, ad_id_queue, demographic_queue
from eyetrack import eye_tracking_thread_func, update_database, extract_number, watching_lock
from dotenv import load_dotenv

from util import get_resource_path


class Context:
    def __init__(self):
        self.face_detection_active = threading.Event()
        self.face_detection_active.set()  # Default: enable face detection
        self.detected_face_queue = queue.Queue(maxsize=1)  # Store only the latest face
        self.ad_text_queue = queue.Queue(maxsize=1)  # Store only the latest ad text
        self.state_lock = threading.Lock()  # Lock for state transitions
        self.eye_tracking_active = threading.Event()
        self.default_video_completed = threading.Event()  # New: signal for default ad playback completion
        self.personalized_video_completed = threading.Event()
        self.eye_tracking_active.clear()
        self.total_watch_time = 0.0


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
            self.context.face_detection_active.set()  # Enable camera usage by the yolo thread upon entering AD Rotation
            # print("[AD R] Set face_detection_active true")
            if self.is_first:
                print("[State] Ad Rotating: Displaying generic ad.")
                self.is_first = False

            if not self.context.detected_face_queue.empty():
                self.context.face_detection_active.clear()  # Pause face detection
                frame, prediction = self.context.detected_face_queue.get()
                # if not self.context.detected_face_queue.empty():
                #     print("not empty!!!!!!!!!!!!!!!!!!!!!!!")
                # if self.context.detected_face_queue.empty():
                #     print("empty!!!!!!!!!!")
                print("[State] LLM Processing: Generating ad text.")
                processing_thread = threading.Thread(target=self.process_frame, args=(prediction,))
                processing_thread.start()
                processing_thread.join()  # Wait for the thread to complete
                self.llm_text_generated_event.wait()  # Wait for ad text generation

                if not self.context.ad_text_queue.empty():
                    # Only query whether to switch to the next state after this video finishes playing
                    self.context.default_video_completed.wait()
                    self.context.default_video_completed.clear()
                    return PersonalizedADDisplaying(self.context)
                else:
                    print("[Error] Ad generation failed, returning to Ad Rotating.")
                    self.context.face_detection_active.set()
                    return self
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
                # print("[ERROR] Queue is full")
                pass  # Discard old data if the queue is full
        else:
            print("[Error] LLM failed outputting")
        self.llm_text_generated_event.set()

class PersonalizedADDisplaying(State):
    def __init__(self, context):
        super().__init__(context)

    def __str__(self):
        return "PersonalizedADDisplaying"

    def handle(self):
        with self.context.state_lock:
            self.context.eye_tracking_active.set()
            print("[State] Eye tracking activated for personalized ad.")


            if not self.context.ad_text_queue.empty():
                debug_ad_text = self.context.ad_text_queue.get_nowait()
                # print(f"[State] Displaying personalized ad with text: {debug_ad_text}")
                # Optionally, send the ad_text to the secondary screen or another system
                # For example: secondary_screen_signal_queue.put(ad_text)
            else:
                print("[Error] No ad text available for personalized ad display")

            self.context.personalized_video_completed.wait()  # Wait for personalized ad playback to complete
            self.context.eye_tracking_active.clear()

            if not ad_id_queue.empty():
                ad_id = extract_number(ad_id_queue.get_nowait())
            else:
                print("[Error] Cannot get ad id")
                exit(1)

            if not demographic_queue.empty():
                prediction = demographic_queue.get_nowait()
            else:
                print("[Error] Cannot get demographic id")
                exit(1)

            with watching_lock:
                watch_time = self.context.total_watch_time
                self.context.total_watch_time = 0

            success = update_database(watch_time, prediction, ad_id)
            if success:
                print(f"[Eyetracking] Database updated: watch_time={watch_time:.2f}, demographics={prediction}, ad_id={ad_id}")
            else:
                print("[Eyetracking] Fatal. Database update failed]")
                exit(1)

            # print("[State] Eye tracking stopped.")
            self.context.personalized_video_completed.clear()  # Reset the signal
            secondary_screen_signal_queue.put("wait")

        return AdRotating(self.context, True)

# Create the main Flask application
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
    env_path = get_resource_path(".env")
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token unfounded, set it in .env field HF_TOKEN")
    pipeline = AdvertisementPipeline(token=token)

    cap_index = int(input("Input camera index (by default 0): "))
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        print("Error: cannot open camera!")
        exit(1)

    from data_integration.user_screen_server import set_context
    set_context(context)

    # Run the Flask app
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

    # Start the CV thread
    cv_thread = threading.Thread(
        target=cv_thread_func,
        args=(cap, context.detected_face_queue, context.face_detection_active)
    )
    cv_thread.daemon = True
    cv_thread.start()

    eye_tracking_thread = threading.Thread(
        target=eye_tracking_thread_func,
        args=(cap, context.eye_tracking_active, context)
    )
    eye_tracking_thread.daemon = True
    eye_tracking_thread.start()

    # Run the state machine
    current_state = AdRotating(context, True)
    # prev_state = str(current_state)
    while True:
        # prev_state = str(current_state)
        current_state = current_state.handle()
        # if (prev_state != str(current_state)):
        #     print("Current state: " + str(current_state) + ", Prev state: " + str(prev_state))
        time.sleep(0.5)  # Control the state machine's pace