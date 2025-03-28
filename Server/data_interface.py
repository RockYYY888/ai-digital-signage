# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
from queue import Queue
from datetime import date

# This is for age_group, gender, race. e.g: ["17-35", "Female", "Asian", "happy"]
secondary_screen_signal_queue = Queue(maxsize=1)

# Provide date for today. e.g: 2025-02-27
today = date.today()
print("Start up. Today's date is:", today)

video_queue = Queue(maxsize=1)
ad_id_queue = Queue(maxsize=1)
demographic_queue = Queue(maxsize=1)

ad_queue = Queue()

frame_queue = Queue(maxsize=1)
