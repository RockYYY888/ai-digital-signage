# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
import time
from Server.data_interface import secondary_screen_signal_queue

# Global variable initialization
last_prediction = None

def generate_test_input():
    global last_prediction
    
    # If there is new data in the queue, update last_detection and last_prediction
    if not secondary_screen_signal_queue.empty():  # Assume there is only one queue prediction_queue
        data = secondary_screen_signal_queue.get()
        if data:
            last_prediction = data

    
    # Return the latest status and data
    if last_prediction:  # If there is a prediction result
        if last_prediction == "wait":
            return {
            "status": "no_face",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
        }
        else:
            return {
            "status": "finished",
            "age": last_prediction[0],
            "gender": last_prediction[1],
            "ethnicity": last_prediction[2],
            "emotion": last_prediction[3],
        }
    else:  # If none
        return {
            "status": "no_face",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
        }

def get_data_stream():
    while True:
        time.sleep(0.1)
        yield generate_test_input()