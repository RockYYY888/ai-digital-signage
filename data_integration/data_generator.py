import time
from data_interface import prediction_queue

last_prediction = None

def generate_test_input():
    global last_prediction
    # If there is new data in the queue, update
    if not prediction_queue.empty():
        last_prediction = prediction_queue.get()
    # Return the latest data or default value
    return list(last_prediction) if last_prediction else ["30-45", "Female", "Asian", "happy"]

def get_data_stream():
    while True:
        yield generate_test_input()
        time.sleep(3)  # Keep the original interval


