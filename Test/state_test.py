# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
import pytest

from unittest.mock import Mock, patch
from state import Context, AdRotating, PersonalizedADDisplaying, ad_id_queue, demographic_queue  

@pytest.mark.timeout(10)
def test_state_transition_logic():
  
    
    
    context = Context()
    
    # Test Case 1: AdRotating to PersonalizedADDisplaying
    initial_state = AdRotating(context, is_first=True)
    
    # Simulate face detection thread putting data
    mock_frame = Mock()
    mock_prediction = ("20-30", "male", "asian")
    context.detected_face_queue.put((mock_frame, mock_prediction))
    
    # Simulate LLM generating ad text
    mock_ad_text = "Buy this cool product!"
    with patch.object(AdRotating, 'process_frame') as mock_process_frame:
        def side_effect(pred):
            context.ad_text_queue.put(mock_ad_text)
            initial_state.llm_text_generated_event.set()
        mock_process_frame.side_effect = side_effect
    
        # Simulate default video completion
        context.default_video_completed.set()
    
        # Execute state transition
        next_state = initial_state.handle()
    
        # Verify state transition
        assert isinstance(next_state, PersonalizedADDisplaying)
        
        # Prepare for PersonalizedADDisplaying's handle method
        context.personalized_video_completed.set() 
        ad_id_queue.put("1.mp4")                   
        demographic_queue.put(mock_prediction)    
        with patch('state.update_database', return_value=True):  # Mock database update
            next_state.handle()  
    
        
        
if __name__ == "__main__":
    pytest.main(["-v"])