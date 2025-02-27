from queue import Queue
from datetime import date

# This is for age_group, gender, race. e.g: ["17-35", "Female", "Asian", "happy"]
prediction_queue = Queue()

# Provide date for today
today = date.today()
print("Today's date is:", today)

