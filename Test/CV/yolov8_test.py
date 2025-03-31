# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
# test/test_yolov8.py

import pytest  
import numpy as np  
from unittest.mock import patch, Mock  
from PIL import Image  
import cv2  

# Fixture create
@pytest.fixture
def mock_frame():
    """Create a mock frame for testing.

    Returns:
        numpy.ndarray: A 100x100x3 black image array with uint8 dtype.
    """
    # create image
    return np.zeros((100, 100, 3), dtype=np.uint8)  # Generate a 100x100 RGB black image

# test analyze_frame（success）
@patch('CV.yolov8.face_detector')  # mock YOLO face_detector function
def test_analyze_frame_face_detected(mock_face_detector, mock_frame):
    """Test analyze_frame when a face is detected.

    This test verifies that analyze_frame returns a prediction and a cropped PIL Image
    when the YOLO face detector identifies a face in the frame.
    """
    # mock YOLO result
    mock_result = Mock()  # Create a mock result object
    mock_box = Mock(xyxy=[[10, 20, 30, 40]])  # Mock bounding box with coordinates [x_min, y_min, x_max, y_max]
    mock_result.boxes = mock_box  # Assign mock bounding box to result
    mock_face_detector.return_value = [mock_result]  # Configure mock face_detector to return a detection

    # import analyze_frame
    from CV.yolov8 import analyze_frame  # Import the function to test

    # use analyze_frame
    prediction, cropped_image = analyze_frame(mock_frame)  # Call analyze_frame with mock frame

    # validation
    assert prediction is not None, "detected face"  # Ensure prediction is returned when face is detected
    assert isinstance(cropped_image, Image.Image), "cut PIL.Image "  # Verify cropped_image is a PIL Image object

# test analyze_frame
@patch('CV.yolov8.face_detector')  # mock YOLO face_detector function
def test_analyze_frame_no_face_detected(mock_face_detector, mock_frame):
    """Test analyze_frame when no face is detected.

    This test verifies that analyze_frame returns None for both prediction and cropped image
    when the YOLO face detector does not identify any faces in the frame.
    """
    # undetected face
    mock_face_detector.return_value = []  # Configure mock face_detector to return an empty list (no detections)

    # import analyze_frame
    from CV.yolov8 import analyze_frame  # Import the function to test

    # use analyze_frame
    prediction, cropped_image = analyze_frame(mock_frame)  # Call analyze_frame with mock frame

    # validation
    assert prediction is None, "undected, None"  # Ensure prediction is None when no face is detected
    assert cropped_image is None, "cut None"  # Ensure cropped_image is None when no face is detected

if __name__ == "__main__":
    pytest.main(["-v"])  # Run pytest with verbose output when script is executed directly