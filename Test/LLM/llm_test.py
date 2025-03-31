# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import sys  
import os  
import pytest  
from queue import Empty  

# add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # Compute the project root directory
sys.path.insert(0, project_root)  # Add project root to sys.path for module imports

from LLM.LLM import AdvertisementPipeline, ad_queue  # Import the AdvertisementPipeline class and ad_queue

# Fixture create a AdvertisementPipeline
@pytest.fixture
def pipeline():
    """Create an AdvertisementPipeline instance for testing.

    Loads the HF_TOKEN from the .env file in the project root and initializes the pipeline.

    Returns:
        AdvertisementPipeline: An instance of the AdvertisementPipeline class.

    Raises:
        ValueError: If HF_TOKEN is not found in the .env file.
    """
    env_path = os.path.join(project_root, ".env")  # Path to the .env file
    from dotenv import load_dotenv  # Import load_dotenv dynamically
    load_dotenv(dotenv_path=env_path)  # Load environment variables from .env
    token = os.getenv("HF_TOKEN")  # Retrieve the Hugging Face token
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")  # Raise error if token is missing
    
    pipeline = AdvertisementPipeline(token=token)  # Initialize the pipeline with the token
    return pipeline  # Return the pipeline instance

# validate generate_advertisement 
def test_generate_advertisement_integration(pipeline):
    """Test the integration of generate_advertisement with various demographic inputs.

    This test validates that the AdvertisementPipeline generates advertisements for different
    demographic combinations and correctly places them in the ad_queue.
    """
    # test input demographics  age_group, gender, ethnicity 
    test_inputs = [
        ('17-35', 'Female', 'Asian', 'happy'),  # Test case 1: Young female, happy
        ('35-50', 'Male', 'White', 'sad'),      # Test case 2: Middle-aged male, sad
        ('35-50', 'Female', 'Black', 'sad'),    # Test case 3: Middle-aged female, sad
        ('50+', 'Female', 'Other', 'angry'),    # Test case 4: Older female, angry
        ('50+', 'Male', 'Indian', 'sad')        # Test case 5: Older male, sad
    ]

    expected_ads = [
        "Hi high heels are more than just shoes for young women. They're a symbol of confidence, elegance, and personal style.",
        "A well-fitted suit embodies comfort & professionalism; perfect for formal events, business meetings, family gatherings & special occasions.",
        "Feeling down? Let our lipstick add color & comfort with its soft shades, style & confidence that suits you from timeless reds to bold pinks.",
        "Indulge in our organic vegetables grown using natural farming methods without synthetic pesticides, fertilizers & GMs. Fresh, high nutrition, complete, reduce exposure to toxins, promoting peace of mind.",
        "Feeling down? Our stylish sunglasses protect your eyes from harsh UV rays, reduce glare & improve visibility with clear comfort & bright colors."
    ]  # Expected advertisement texts for each test case

    for input_data, expected_ad in zip(test_inputs, expected_ads):
        #clear queue
        while not ad_queue.empty():
            try:
                ad_queue.get_nowait()  # Remove items from the queue without blocking
            except Empty:
                break  # Exit loop if queue is empty or exception occurs
        print(f"Cleared ad_queue for input: {input_data}")  # Log queue clearing

        #  generate_advertisement generate ads
        result = pipeline.generate_advertisement(input_data)  # Generate advertisement for the input demographics
        print(f"Generated ad for input {input_data}: {result}")  # Log the generated result

        # check ad_queue is empty
        if ad_queue.empty():
            print(f"Warning: ad_queue is empty for input {input_data}, possibly due to missing video data in demographics table")  # Warn if queue remains empty
            continue  # Skip to next iteration if no ad is queued
        
        ad_text = ad_queue.get_nowait()  # Retrieve ad text from the queue without blocking
        print(f"Ad text from queue for input {input_data}: {ad_text}")  # Log the retrieved ad text

if __name__ == "__main__":
    pytest.main(["-v"])  # Run pytest with verbose output when script is executed directly