# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import pytest  # For writing and running unit tests
from unittest.mock import Mock, patch  # For mocking objects and patching methods
from state import Context, AdRotating, PersonalizedADDisplaying, ad_id_queue, demographic_queue  # Import state-related classes and queues

@pytest.mark.timeout(10)  # Set a 10-second timeout for the test to prevent hangs
def test_state_transition_logic():
    """Test the state transition logic from AdRotating to PersonalizedADDisplaying.

    This test verifies that the AdRotating state transitions correctly to PersonalizedADDisplaying
    when a face is detected, video completes, and LLM generates ad text. It also tests the
    subsequent handling in PersonalizedADDisplaying.
    """
    
    # Initialize the Context object to manage state transitions
    context = Context()
    
    # Test Case 1: AdRotating to PersonalizedADDisplaying
    initial_state = AdRotating(context, is_first=True)  # Create initial AdRotating state
    
    # Simulate face detection thread putting data
    mock_frame = Mock()  # Mock a video frame object
    mock_prediction = ("20-30", "male", "asian")  # Mock demographic prediction (age, gender, ethnicity)
    context.detected_face_queue.put((mock_frame, mock_prediction))  # Add mock data to detected face queue
    
    # Simulate LLM generating ad text
    mock_ad_text = "Buy this cool product!"  # Mock advertisement text
    with patch.object(AdRotating, 'process_frame') as mock_process_frame:  # Patch the process_frame method
        def side_effect(pred):
            # Define side effect to simulate LLM text generation
            context.ad_text_queue.put(mock_ad_text)  # Add mock ad text to queue
            initial_state.llm_text_generated_event.set()  # Signal LLM text generation completion
        mock_process_frame.side_effect = side_effect  # Assign side effect to mocked method
    
        # Simulate default video completion
        context.default_video_completed.set()  # Set event to indicate default video has finished
    
        # Execute state transition
        next_state = initial_state.handle()  # Trigger the handle method to transition state
    
        # Verify state transition
        assert isinstance(next_state, PersonalizedADDisplaying)  # Check that next state is PersonalizedADDisplaying
        
        # Prepare for PersonalizedADDisplaying's handle method
        context.personalized_video_completed.set()  # Set event to indicate personalized video completion
        ad_id_queue.put("1.mp4")  # Add mock ad ID to the ad ID queue
        demographic_queue.put(mock_prediction)  # Add mock demographic data to the demographic queue
        with patch('state.update_database', return_value=True):  # Mock the update_database function to return True
            next_state.handle()  # Execute the handle method of PersonalizedADDisplaying state
    
if __name__ == "__main__":
    pytest.main(["-v"])  # Run pytest with verbose output when script is executed directly