# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
import pytest
import threading
import time
import numpy as np
from unittest.mock import Mock, patch
from eyetrack import eye_tracking_thread_func

class Context:
    def __init__(self):
        self.total_watch_time = 0.0
@pytest.mark.timeout(10)
def test_timing_logic():
    # --- Test Case 1: Watching (face detected, eyes open) ---
    context = Context()
    active_flag = threading.Event()
    active_flag.set()
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:
        mock_face = Mock()
        mock_detector.return_value.return_value = [mock_face]
        with patch('dlib.shape_predictor') as mock_predictor:
            left_eye = Mock(x=50, y=0)
            right_eye = Mock(x=50, y=0)
            landmarks = Mock()
            landmarks.part.side_effect = [left_eye, right_eye]
            mock_predictor.return_value = landmarks
            with patch('time.time') as mock_time:
                def time_generator():
                    t = 0.0
                    while True:
                        yield t
                        t += 0.03
                mock_time.side_effect = time_generator()

                thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))
                thread.start()
                time.sleep(2)
                active_flag.clear()
                thread.join(timeout=1)

                print(f"Test Case 1: total_watch_time = {context.total_watch_time}")
                assert context.total_watch_time < 2.2, (
                    f"Expected total_watch_time around 2s, got {context.total_watch_time}"
                )

    # --- Test Case 2: Not Watching (no face detected) ---
    context.total_watch_time = 0.0
    active_flag.set()
    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:
        mock_detector.return_value.return_value = []
        with patch('time.time') as mock_time:
            mock_time.side_effect = time_generator()
            thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))
            thread.start()
            time.sleep(2)
            active_flag.clear()
            thread.join(timeout=1)
            print(f"Test Case 2: total_watch_time = {context.total_watch_time}")
            assert context.total_watch_time == 0.0, (
                f"Expected total_watch_time to be 0 when no face detected, got {context.total_watch_time}"
            )

    # --- Test Case 3: Thread Inactive ---
    context.total_watch_time = 0.0
    active_flag.clear()
    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:
        mock_face = Mock()
        mock_detector.return_value.return_value = [mock_face]
        with patch('dlib.shape_predictor') as mock_predictor:
            left_eye = Mock(x=0, y=0)
            right_eye = Mock(x=20, y=0)
            landmarks = Mock()
            landmarks.part.side_effect = [left_eye, right_eye]
            mock_predictor.return_value = landmarks
            with patch('time.time') as mock_time:
                mock_time.side_effect = time_generator()
                thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))
                thread.start()
                time.sleep(2)
                thread.join(timeout=1)
                print(f"Test Case 3: total_watch_time = {context.total_watch_time}")
                assert context.total_watch_time == 0.0, (
                    f"Expected total_watch_time to be 0 when thread inactive, got {context.total_watch_time}"
                )
 
if __name__ == "__main__":
    pytest.main(["-v"])               