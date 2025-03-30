# test/test_yolov8.py
import pytest
import numpy as np
from unittest.mock import patch, Mock
from PIL import Image
import cv2

# Fixture create
@pytest.fixture
def mock_frame():
    # create image
    return np.zeros((100, 100, 3), dtype=np.uint8)

# test analyze_frame（success）
@patch('CV.yolov8.face_detector')  # mock YOLO 
def test_analyze_frame_face_detected(mock_face_detector, mock_frame):
    # mock YOLO result
    mock_result = Mock()
    mock_box = Mock(xyxy=[[10, 20, 30, 40]])  # edge
    mock_result.boxes = mock_box
    mock_face_detector.return_value = [mock_result]

    # import analyze_frame
    from CV.yolov8 import analyze_frame

    #  use analyze_frame
    prediction, cropped_image = analyze_frame(mock_frame)

    # validation
    assert prediction is not None, "detected face"
    assert isinstance(cropped_image, Image.Image), "cut PIL.Image "

# test analyze_frame
@patch('CV.yolov8.face_detector')  # mock YOLO 
def test_analyze_frame_no_face_detected(mock_face_detector, mock_frame):
    # undetected face
    mock_face_detector.return_value = []

    # import analyze_frame
    from CV.yolov8 import analyze_frame

    # use analyze_frame
    prediction, cropped_image = analyze_frame(mock_frame)

    # validation
    assert prediction is None, "undected, None"
    assert cropped_image is None, "cut None"

if __name__ == "__main__":
    pytest.main(["-v"])