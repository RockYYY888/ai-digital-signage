# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
"""Module for defining data queues and utilities for the Targeted Digital Signage system.

This module initializes various queues for handling signals, video data, advertisements,
demographics, and frames, along with providing the current date for logging purposes.
"""

from queue import Queue
from datetime import date

# This is for age_group, gender, race. e.g: ["17-35", "Female", "Asian", "happy"]
secondary_screen_signal_queue = Queue(maxsize=1)
"""Queue: Stores signals for the secondary screen, containing demographic and emotion data.

The queue has a maximum size of 1 and expects tuples or lists in the format
[age_group, gender, race, emotion], e.g., ["17-35", "Female", "Asian", "happy"].
"""

# Provide date for today. e.g: 2025-02-27
today = date.today()
"""date: The current date, used for logging or display purposes.

Initialized at startup to reflect the system's current date.
"""
print("Start up. Today's date is:", today)

video_queue = Queue(maxsize=1)
"""Queue: Stores video file information for display.

The queue has a maximum size of 1 and holds video-related data, such as file names.
"""

ad_id_queue = Queue(maxsize=1)
"""Queue: Stores advertisement IDs for tracking.

The queue has a maximum size of 1 and holds unique identifiers for advertisements.
"""

demographic_queue = Queue(maxsize=1)
"""Queue: Stores demographic data for targeting advertisements.

The queue has a maximum size of 1 and holds tuples of demographic info (e.g., age, gender, race).
"""

ad_queue = Queue()
"""Queue: Stores generated advertisement text.

The queue has no size limit and holds advertisement content for display.
"""

frame_queue = Queue(maxsize=1)
"""Queue: Stores image frames for face detection display.

The queue has a maximum size of 1 and holds image data for processing.
"""