import time
from data_integration.data_interface import secendary_screen_signal_queue

# Global variable initialization
last_prediction = None

def generate_test_input():
    global last_prediction
    
    # If there is new data in the queue, update last_detection and last_prediction
    if not secendary_screen_signal_queue.empty():  # Assume there is only one queue prediction_queue
        data = secendary_screen_signal_queue.get()
        if data == "no_face":
            last_prediction = data

    
    # Return the latest status and data
    if last_prediction == "no_face":  # If there is a prediction result
        return {
            "status": "no_face",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
        }
    elif last_prediction:  # If none
        return {
            "status": "finished",
            "age": last_prediction[0],
            "gender": last_prediction[1],
            "ethnicity": last_prediction[2],
            "emotion": last_prediction[3],
        }

def get_data_stream():
    while True:
        time.sleep(0.1)
        yield generate_test_input()