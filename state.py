# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import os
import random
import threading
import time
import queue
import webbrowser
import cv2
from flask import Flask
from CV.yolov8 import cv_thread_func
from LLM.LLM import AdvertisementPipeline
from Server.server import secondary_screen_app
from Server.user_screen_server import user_screen
from Dashboard.dashboard import init_dashboard
from Server.data_interface import secondary_screen_signal_queue, ad_id_queue, demographic_queue
from eyetracking.eyetrack import eye_tracking_thread_func, update_database, extract_number, watching_lock
from dotenv import load_dotenv
from multiprocessing import Process

from util import get_resource_path


class Context:
    """Manages shared state and synchronization primitives for the Targeted Digital Signage system.

    This class encapsulates the state and threading-related objects used across different components
    of the system, such as face detection, eye tracking, and advertisement display.

    Attributes:
        face_detection_active (threading.Event): Controls whether face detection is active.
        user_screen_focus (threading.Event): Indicates if the user screen is in focus.
        detected_face_queue (queue.Queue): Stores the latest detected face data (max size 1).
        ad_text_queue (queue.Queue): Stores the latest generated advertisement text (max size 1).
        state_lock (threading.Lock): Synchronizes state transitions.
        eye_tracking_active (threading.Event): Controls whether eye tracking is active.
        default_video_completed (threading.Event): Signals completion of default ad playback.
        personalized_video_completed (threading.Event): Signals completion of personalized ad playback.
        total_watch_time (float): Accumulates total watch time for eye tracking.
    """

    def __init__(self):
        """Initialize the Context with default settings and synchronization objects."""
        self.face_detection_active = threading.Event()
        self.user_screen_focus = threading.Event()
        self.user_screen_focus.clear()
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
    """Base class for state machine states in the Targeted Digital Signage system.

    Args:
        context (Context): The shared context object for state management.
        is_first (bool, optional): Indicates if this is the initial state. Defaults to False.
    """

    def __init__(self, context, is_first=False):
        self.context = context
        self.is_first = is_first

    def handle(self):
        """Handle the logic for this state and return the next state.

        Returns:
            State: The next state in the state machine.
        """
        pass


class AdRotating(State):
    """State for rotating generic advertisements until a face is detected.

    Attributes:
        llm_text_generated_event (threading.Event): Signals when ad text generation is complete.
    """

    def __init__(self, context, is_first=False):
        """Initialize the AdRotating state."""
        super().__init__(context)
        self.is_first = is_first
        self.llm_text_generated_event = threading.Event()

    def __str__(self):
        """Return a string representation of the state."""
        return "ADRotation"

    def handle(self):
        """Execute the AdRotating state logic.

        Enables face detection and waits for a face to be detected. If a face is found, generates
        personalized ad text and transitions to the PersonalizedADDisplaying state.

        Returns:
            State: The next state (self or PersonalizedADDisplaying).
        """
        with self.context.state_lock:
            self.context.face_detection_active.set()  # Enable camera usage by the yolo thread
            if self.is_first:
                print("[State] Ad Rotating: Displaying generic ad.")
                self.is_first = False

            if not self.context.detected_face_queue.empty():
                self.context.face_detection_active.clear()  # Pause face detection after any face detected
                frame, prediction = self.context.detected_face_queue.get()
                print("[State] LLM Processing: Generating ad text.")
                processing_thread = threading.Thread(target=self.process_frame, args=(prediction,))
                processing_thread.start()
                processing_thread.join()  # Wait for the thread to complete
                self.llm_text_generated_event.wait()  # Wait for ad text generation

                if not self.context.ad_text_queue.empty():
                    self.context.default_video_completed.wait()
                    self.context.default_video_completed.clear()
                    return PersonalizedADDisplaying(self.context)
                else:
                    print("[Error] Ad generation failed, returning to Ad Rotating.")
                    self.context.face_detection_active.set()
                    return self
            else:
                return self

    def process_frame(self, prediction):
        """Generate advertisement text based on face prediction and store it in the queue.

        Args:
            prediction: The demographic or face data used to generate the ad text.
        """
        ad_text = pipeline.generate_advertisement(prediction)
        if ad_text:
            try:
                self.context.ad_text_queue.put_nowait(ad_text)
            except queue.Full:
                pass  # Discard old data if the queue is full
        else:
            print("[Error] LLM failed outputting")
        self.llm_text_generated_event.set()


class PersonalizedADDisplaying(State):
    """State for displaying personalized advertisements with eye tracking."""

    def __init__(self, context):
        """Initialize the PersonalizedADDisplaying state."""
        super().__init__(context)

    def __str__(self):
        """Return a string representation of the state."""
        return "PersonalizedADDisplaying"

    def handle(self):
        """Execute the PersonalizedADDisplaying state logic.

        Activates eye tracking, displays a personalized ad, updates the database with watch time,
        and transitions back to AdRotating.

        Returns:
            State: The next state (AdRotating).
        """
        with self.context.state_lock:
            self.context.eye_tracking_active.set()
            print("[State] Eye tracking activated for personalized ad.")

            # if not self.context.ad_text_queue.empty():
            #     debug_ad_text = self.context.ad_text_queue.get_nowait()
            # else:
            #     print("[Error] No ad text available for personalized ad display")

            self.context.personalized_video_completed.wait()  # Wait for personalized ad playback
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
    """Serve the root endpoint with a welcome message and available routes.

    Returns:
        str: HTML string listing available endpoints.
    """
    return """
    <h2>Welcome to the Flask Server!</h2>
    <p>Available Endpoints:</p>
    <ul>
        <li><a href="/user-screen/">User Screen</a></li>
        <li><a href="/secondary-screen/">Secondary Screen</a></li>
        <li><a href="/dashboard/">Dashboard</a></li>
    </ul>
    """


# Define the range of usable ports
MIN_PORT = 5000
MAX_PORT = 8000

# Set the seed to the current timestamp in seconds
current_time = int(time.time())  # Unix timestamp in seconds
random.seed(current_time)
random_port = random.randint(MIN_PORT, MAX_PORT)

if __name__ == "__main__":
    """Main entry point for the Targeted Digital Signage application.

    Initializes the context, loads environment variables, sets up the camera, starts the Flask server,
    and runs the state machine and supporting threads.
    """
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

    from Server.user_screen_server import set_context
    set_context(context)

    # Run the Flask app
    flask_thread = threading.Thread(target=main_app.run, kwargs={
        "host": "127.0.0.1",
        "port": random_port,
        "threaded": True,
        "debug": False
    })
    flask_thread.daemon = True
    flask_thread.start()

    webbrowser.open(f"http://127.0.0.1:{random_port}/user-screen/")
    webbrowser.open(f"http://127.0.0.1:{random_port}/secondary-screen/")
    webbrowser.open(f"http://127.0.1:{random_port}/dashboard/")

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
    while True:
        current_state = current_state.handle()
        time.sleep(0.5)  # Control the state machine's pace