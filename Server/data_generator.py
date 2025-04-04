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
    """Generate test input based on the latest prediction data.

    This function retrieves the latest data from the secondary screen signal queue,
    updates the global last_prediction variable, and returns the corresponding status
    and attributes.

    Returns:
        dict: A dictionary containing status, age, gender, ethnicity, and emotion.
    """
    global last_prediction
    
    # If there is new data in the queue, update last_prediction
    if not secondary_screen_signal_queue.empty():
        data = secondary_screen_signal_queue.get()
        if data:
            last_prediction = data

    # Return the latest status and data
    if last_prediction:
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
    else:
        return {
            "status": "no_face",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
        }

def get_data_stream():
    """Continuously generate and yield test input data.

    This function runs an infinite loop, sleeping for a short duration,
    and then yielding the latest test input generated by `generate_test_input`.

    Yields:
        dict: The latest test input data.
    """
    while True:
        time.sleep(0.1)
        yield generate_test_input()
