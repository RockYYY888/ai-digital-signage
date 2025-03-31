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
    """A mock context class for tracking watch time in eye tracking tests."""
    def __init__(self):
        self.total_watch_time = 0.0  # Initialize total watch time to zero

@pytest.mark.timeout(10)  # Set a 10-second timeout for the test to prevent hangs
def test_timing_logic():
    """Test the timing logic of the eye tracking thread under various conditions.

    This test verifies the behavior of eye_tracking_thread_func in three scenarios:
    1. Face detected with eyes open (watch time accumulates).
    2. No face detected (watch time remains zero).
    3. Thread inactive (watch time remains zero).
    """
    # --- Test Case 1: Watching (face detected, eyes open) ---
    context = Context()  # Create a new context instance
    active_flag = threading.Event()  # Create an event to control thread activity
    active_flag.set()  # Set the thread to active state
    cap = Mock()  # Mock a video capture object
    cap.isOpened.return_value = True  # Simulate camera being open
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))  # Mock frame reading with a black image

    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:  # Mock face detector
        mock_face = Mock()  # Mock a detected face
        mock_detector.return_value.return_value = [mock_face]  # Return a list with one face
        with patch('dlib.shape_predictor') as mock_predictor:  # Mock shape predictor for landmarks
            left_eye = Mock(x=50, y=0)  # Mock left eye landmark
            right_eye = Mock(x=50, y=0)  # Mock right eye landmark (eyes aligned vertically)
            landmarks = Mock()  # Mock landmarks object
            landmarks.part.side_effect = [left_eye, right_eye]  # Return eye landmarks sequentially
            mock_predictor.return_value = landmarks  # Configure predictor to return landmarks
            with patch('time.time') as mock_time:  # Mock time.time for controlled timing
                def time_generator():
                    t = 0.0
                    while True:
                        yield t
                        t += 0.03  # Increment time by 0.03 seconds per frame
                mock_time.side_effect = time_generator()  # Use generator to simulate time progression

                thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))  # Create eye tracking thread
                thread.start()  # Start the thread
                time.sleep(2)  # Let the thread run for 2 seconds
                active_flag.clear()  # Signal the thread to stop
                thread.join(timeout=1)  # Wait for thread to finish with a 1-second timeout

                print(f"Test Case 1: total_watch_time = {context.total_watch_time}")  # Log watch time
                assert context.total_watch_time < 2.2, (
                    f"Expected total_watch_time around 2s, got {context.total_watch_time}"
                )  # Verify watch time is close to 2 seconds

    # --- Test Case 2: Not Watching (no face detected) ---
    context.total_watch_time = 0.0  # Reset watch time
    active_flag.set()  # Set thread to active state
    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:  # Mock face detector
        mock_detector.return_value.return_value = []  # Return no faces detected
        with patch('time.time') as mock_time:  # Mock time.time for controlled timing
            mock_time.side_effect = time_generator()  # Reuse time generator
            thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))  # Create eye tracking thread
            thread.start()  # Start the thread
            time.sleep(2)  # Let the thread run for 2 seconds
            active_flag.clear()  # Signal the thread to stop
            thread.join(timeout=1)  # Wait for thread to finish
            print(f"Test Case 2: total_watch_time = {context.total_watch_time}")  # Log watch time
            assert context.total_watch_time == 0.0, (
                f"Expected total_watch_time to be 0 when no face detected, got {context.total_watch_time}"
            )  # Verify watch time remains zero

    # --- Test Case 3: Thread Inactive ---
    context.total_watch_time = 0.0  # Reset watch time
    active_flag.clear()  # Set thread to inactive state
    with patch('dlib.get_frontal_face_detector', return_value=Mock()) as mock_detector:  # Mock face detector
        mock_face = Mock()  # Mock a detected face
        mock_detector.return_value.return_value = [mock_face]  # Return a list with one face
        with patch('dlib.shape_predictor') as mock_predictor:  # Mock shape predictor for landmarks
            left_eye = Mock(x=0, y=0)  # Mock left eye landmark
            right_eye = Mock(x=20, y=0)  # Mock right eye landmark (eyes aligned horizontally)
            landmarks = Mock()  # Mock landmarks object
            landmarks.part.side_effect = [left_eye, right_eye]  # Return eye landmarks sequentially
            mock_predictor.return_value = landmarks  # Configure predictor to return landmarks
            with patch('time.time') as mock_time:  # Mock time.time for controlled timing
                mock_time.side_effect = time_generator()  # Reuse time generator
                thread = threading.Thread(target=eye_tracking_thread_func, args=(cap, active_flag, context))  # Create eye tracking thread
                thread.start()  # Start the thread
                time.sleep(2)  # Let the thread run for 2 seconds (though inactive)
                thread.join(timeout=1)  # Wait for thread to finish
                print(f"Test Case 3: total_watch_time = {context.total_watch_time}")  # Log watch time
                assert context.total_watch_time == 0.0, (
                    f"Expected total_watch_time to be 0 when thread inactive, got {context.total_watch_time}"
                )  # Verify watch time remains zero
 
if __name__ == "__main__":
    pytest.main(["-v"])  # Run pytest with verbose output when script is executed directly