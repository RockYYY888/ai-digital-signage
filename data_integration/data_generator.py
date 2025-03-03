import time
from data_integration.data_interface import prediction_queue,detect_queue

# Global variable initialization
last_prediction = None
last_detection = None
last_feedback = None

def generate_test_input():
    global last_prediction, last_detection, last_feedback
    
    # If there is new data in the queue, update last_detection and last_prediction
    if not prediction_queue.empty():  # Assume there is only one queue prediction_queue
        data = prediction_queue.get()
        if data == ("analyzing"):  # Indicates that a face was detected but not analyzed
            last_detection = data
            last_prediction = None
            last_feedback = None
        elif data == ("no_face"):  # Indicates that no face was detected
            last_detection = None
            last_prediction = None
            last_feedback = None
        elif data == ("feedback"): # Waiting for feedback
            last_detection = None
            last_prediction = None
            last_feedback = data
        else:  # Indicates that the analysis is complete
            last_prediction = data
            last_detection = None
            last_feedback = None
    
    # Return the latest status and data
    if last_prediction:  # If there is a prediction result
        return {
            "status": "finished",
            "age": last_prediction[0],
            "gender": last_prediction[1],
            "ethnicity": last_prediction[2],
            "emotion": last_prediction[3],
        }
    elif last_detection:  # If a face is detected but not predicted
        return {
            "status": "detected",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
        }
    elif last_feedback:   # If you are waiting for feedback
        return {
            "status": "feedback",
            "age": "",
            "gender": "",
            "ethnicity": "",
            "emotion": "",
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