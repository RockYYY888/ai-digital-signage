from queue import Queue
from datetime import date

# This is for age_group, gender, race. e.g: ["17-35", "Female", "Asian", "happy"]
prediction_queue = Queue()

# This is for check the state of whether face is detected
detect_queue = Queue()

# Provide date for today. e.g: 2025-02-27
today = date.today()
print("Start up. Today's date is:", today)

# Get product name to deliver to dashboard. e.g: Soda
product_queue = Queue()
