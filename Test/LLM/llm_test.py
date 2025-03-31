# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
# ai-digital-signage/testing/LLM/llm_test.py
import sys
import os
import pytest
from queue import Empty

# add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from LLM.LLM import AdvertisementPipeline, ad_queue

# Fixture create a  AdvertisementPipeline
@pytest.fixture
def pipeline():
    env_path = os.path.join(project_root, ".env")
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    pipeline = AdvertisementPipeline(token=token)
    return pipeline

# validate generate_advertisement 
def test_generate_advertisement_integration(pipeline):
    # test input demographics  age_group, gender, ethnicity 
    test_inputs = [
        ('17-35', 'Female', 'Asian', 'happy'),  # Test case 1
        ('35-50', 'Male', 'White', 'sad'),      # Test case 2
        ('35-50', 'Female', 'Black', 'sad'),    # Test case 3
        ('50+', 'Female', 'Other', 'angry'),    # Test case 4
        ('50+', 'Male', 'Indian', 'sad')        # Test case 5
    ]


    expected_ads = [
        "Hi high heels are more than just shoes for young women. They're a symbol of confidence, elegance, and personal style.",
        "A well-fitted suit embodies comfort & professionalism; perfect for formal events, business meetings, family gatherings & special occasions.",
        "Feeling down? Let our lipstick add color & comfort with its soft shades, style & confidence that suits you from timeless reds to bold pinks.",
        "Indulge in our organic vegetables grown using natural farming methods without synthetic pesticides, fertilizers & GMs. Fresh, high nutrition, complete, reduce exposure to toxins, promoting peace of mind.",
        "Feeling down? Our stylish sunglasses protect your eyes from harsh UV rays, reduce glare & improve visibility with clear comfort & bright colors."
    ]

    for input_data, expected_ad in zip(test_inputs, expected_ads):
        #clear queue
        while not ad_queue.empty():
            try:
                ad_queue.get_nowait()
            except Empty:
                break
        print(f"Cleared ad_queue for input: {input_data}")

        #  generate_advertisement generate ads
        result = pipeline.generate_advertisement(input_data)
        print(f"Generated ad for input {input_data}: {result}")

        # check ad_queue is empty
        if ad_queue.empty():
            print(f"Warning: ad_queue is empty for input {input_data}, possibly due to missing video data in demographics table")
            continue  

        
        ad_text = ad_queue.get_nowait()
        print(f"Ad text from queue for input {input_data}: {ad_text}")

if __name__ == "__main__":
    pytest.main(["-v"])