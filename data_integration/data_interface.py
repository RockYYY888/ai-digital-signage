from queue import Queue
from datetime import date

# This is for age_group, gender, race. e.g: ["17-35", "Female", "Asian", "happy"]
secendary_screen_signal_queue = Queue(maxsize=1)

# Provide date for today. e.g: 2025-02-27
today = date.today()
print("Start up. Today's date is:", today)

video_queue = Queue()

ad_queue = Queue()

frame_queue = Queue(maxsize=1)
