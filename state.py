# state.py
import threading
import time
import queue
from CV.yolov8 import cv_thread_func, analyze_frame
from LLM.LLM import AdvertisementPipeline
from data_integration.server import app

class Context:
    def __init__(self):
        self.face_detection_active = threading.Event()
        self.face_detection_active.set()  # Enable face detection by default
        self.detected_face_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.ad_text_queue = queue.Queue()
        self.current_ad_text = None

class State:
    def __init__(self, context):
        self.context = context

    def handle(self):
        pass

class AdRotating(State):
    def handle(self):
        print("Ad Rotating: Displaying generic ad.")
        try:
            frame, prediction = self.context.detected_face_queue.get(timeout=5)  # Wait up to 5 seconds
            self.context.face_detection_active.clear()  # Pause face detection
            return ModelProcessing(self.context, frame, prediction)
        except queue.Empty:
            return self  # If no face detected, stay in AdRotating

class ModelProcessing(State):
    def __init__(self, context, frame, prediction):
        super().__init__(context)
        self.frame = frame
        self.prediction = prediction
        self.ad_generated_event = threading.Event()  # Add an event to signal completion

    def handle(self):
        print("Model Processing: Generating ad text.")
        processing_thread = threading.Thread(target=self.process_frame)
        processing_thread.start()
        processing_thread.join()
        self.ad_generated_event.wait()  # Wait for the ad text to be generated
        if self.context.ad_text_queue.empty():
            print("Ad generation failed, returning to Ad Rotating.")
            self.context.face_detection_active.set()
            return AdRotating(self.context)
        return PersonalizedADDisplaying(self.context)

    def process_frame(self):
        if self.prediction is None:
            self.ad_generated_event.set()  # Signal even if no prediction
            return
        ad_text = pipeline.generate_advertisement(self.prediction)
        if ad_text:
            self.context.ad_text_queue.put(ad_text)
        self.ad_generated_event.set()  # Signal that processing is done

class PersonalizedADDisplaying(State):
    def handle(self):
        ad_text = self.context.ad_text_queue.get()
        self.context.current_ad_text = ad_text
        print(f"Personalized AD Displaying: {ad_text}")
        time.sleep(10)
        return FeedbackCollecting(self.context)

class FeedbackCollecting(State):
    def handle(self):
        print("Feedback Collecting: Displaying QR code.")
        self.context.face_detection_active.set()  # Resume face detection
        start_time = time.time()
        feedback_duration = 10
        while time.time() - start_time < feedback_duration:
            if not self.context.detected_face_queue.empty():
                frame, prediction = self.context.detected_face_queue.get()
                return ModelProcessing(self.context, frame, prediction)
            time.sleep(1)
        return AdRotating(self.context)

if __name__ == "__main__":
    context = Context()
    pipeline = AdvertisementPipeline()

    # Start Flask thread
    flask_thread = threading.Thread(
        target=app.run,
        kwargs={'threaded': True, 'port': 5000}
    )
    flask_thread.daemon = True
    flask_thread.start()

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
        time.sleep(0.1)
