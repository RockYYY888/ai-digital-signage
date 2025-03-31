# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from ad_pool.video_selection import *
import random
import re
from functools import lru_cache
from Server.data_interface import ad_queue, video_queue, ad_id_queue, demographic_queue
from util import get_resource_path

# Load the model and tokenizer globally
model_name = "meta-llama/Llama-3.2-1B-Instruct"

class AdvertisementGenerator:
    """Class to encapsulate advertisement generation functionality"""

    EMOTION_TONE_MAP = {
        "sad": "empathetic and comforting",
        "angry": "assertive yet respectful",
        "happy": "energetic and enthusiastic",
        "neutral": "professional and trustworthy"
    }

    GENERATION_PARAMS = {
        "max_new_tokens": 60,
        "min_new_tokens": 10,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "num_beams": 1,
    }

    def __init__(self, token):
        self.tokenizer = None
        self.model = None
        self.token = token
        self.load_model_and_tokenizer()

    @lru_cache(maxsize=1)
    def load_model_and_tokenizer(self):
        """Load the transformer model and tokenizer using the provided token.

        Raises:
            RuntimeError: If model or tokenizer loading fails.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                token=self.token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                token=self.token
            )

            # Configure tokenizer special tokens
            self.configure_special_tokens()

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def configure_special_tokens(self):
        """Configure special tokens for the tokenizer and model."""
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if not self.tokenizer.eos_token:
            self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def build_system_prompt(self):
        """Build the system prompt for the language model.

        Returns:
            str: The constructed system prompt.
        """
        return (
            "You are a skilled copywriter at a global ad agency. Your only job is to deliver ad content matching my exact needs. "
            "Output must be in double quotes, strictly 20-30 words, no more, no less. "
            "If the text exceeds 30 words or falls below 20, adjust it before outputting. "
            "No extra text or comments allowed—only the ad content in quotes."
        )

    def build_user_prompt(self, demographics, product, context):
        """Construct the user prompt for ad generation based on input parameters.

        Args:
            demographics (dict): Dictionary containing 'race', 'gender', 'age_range', and 'emotion'.
            product (str): The product name to advertise.
            context (list): List of strings describing the context.

        Returns:
            str: The constructed user prompt.
        """
        tone = self.EMOTION_TONE_MAP.get(demographics['emotion'], "professional")
        context_text = " ".join(context)
        product_name = product

        return (
            f"Write a one-sentence creative ad text for {product_name}, strictly 20-30 words. "
            f"Target: {demographics['race']} {demographics['gender']}, aged {demographics['age_range']}, feeling {demographics['emotion']}. "
            f"Use a {tone} tone. Highlight unique features. Background: {context_text}. "
            f"Ensure 20-30 words, adjust if needed, and return only the ad content in quotes."
        )

    def construct_messages(self, demographics, product, context):
        """Construct the message structure for the language model input.

        Args:
            demographics (dict): Dictionary containing 'race', 'gender', 'age_range', and 'emotion'.
            product (str): The product name to advertise.
            context (list): List of strings describing the context.

        Returns:
            list: List of dictionaries representing the system and user prompts.
        """
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(demographics, product, context)},
        ]

    def generate_ad_text(self, messages):
        """Generate advertisement text using the language model.

        Args:
            messages (list): List of dictionaries containing the system and user prompts.

        Returns:
            str: The generated advertisement text.

        Raises:
            RuntimeError: If text generation fails.
        """
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )

            outputs = self.model.generate(
                inputs,
                **self.GENERATION_PARAMS,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return self.process_output(outputs)

        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

    def process_output(self, outputs):
        """Process and clean the model's output.

        Args:
            outputs: The raw output from the language model.

        Returns:
            str: The cleaned advertisement text.
        """
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return self.extract_ad_content(response)

    def extract_ad_content(self, text):
        """Extract the advertisement content between quotes from the text.

        Args:
            text (str): The raw text output from the model.

        Returns:
            str: The extracted advertisement content, or the stripped text if no quotes are found.
        """
        match = re.search(r'"([^"]+)"', text)
        return match.group(1).strip() if match else text.strip()

class AdvertisementPipeline:
    """Orchestration class for ad generation pipeline"""

    def __init__(self, token):
        self.generator = AdvertisementGenerator(token)  # deliver token
        self.debug_mode = False

    def parse_demographics(self, input_data):
        """Parse and validate demographic input data.

        Args:
            input_data (tuple): Tuple containing (age_range, gender, race, emotion).

        Returns:
            dict: Structured demographic data.

        Raises:
            ValueError: If input_data does not contain exactly 4 elements.
        """
        if len(input_data) != 4:
            raise ValueError("Invalid input format. Expected (age_range, gender, race, emotion)")

        demographics = input_data[0], input_data[1], input_data[2]
        demographic_queue.put(demographics)

        return {
            'age_range': input_data[0],
            'gender': input_data[1],
            'race': input_data[2],
            'emotion': input_data[3]
        }

    def select_video(self, demographics):
        """Select a video based on demographic criteria.

        Args:
            demographics (dict): Dictionary containing 'age_range', 'gender', and 'race'.

        Returns:
            dict: Information about the selected video including file name, description, weight, and product.

        Raises:
            ValueError: If no matching videos are found.
        """
        videos = get_targeted_videos_with_ads(
            demographics["age_range"],
            demographics["gender"],
            demographics["race"]
        )

        if not videos:
            raise ValueError("No matching videos found")

        video_files = [video[0] for video in videos]
        weights = [video[2] for video in videos]

        selected_video = random.choices(video_files, weights=weights, k=1)[0]
        selected = next(video for video in videos if video[0] == selected_video)

        video_queue.put(selected[0])
        ad_id_queue.put(selected[0])

        return {
            'file_name': selected[0],
            'description': selected[1],
            'weight': selected[2],
            'product': selected[3]
        }

    def print_debug_info(self, video_info):
        """Print debug information about the selected video if debug mode is enabled.

        Args:
            video_info (dict): Dictionary containing video details (file_name, description, weight, product).
        """
        if self.debug_mode:
            print(f"Selected Video: {video_info['file_name']}")
            print(f"Description: {video_info['description']}")
            print(f"Weight: {video_info['weight']}")
            print(f"Product: {video_info['product']}\n")

    def generate_advertisement(self, input_data):
        """Generate an advertisement based on input demographics.

        Args:
            input_data (tuple): Tuple containing (age_range, gender, race, emotion).

        Returns:
            str: The generated advertisement text, or None if an error occurs.
        """
        try:
            demographics = self.parse_demographics(input_data)
            video_info = self.select_video(demographics)
            messages = self.generator.construct_messages(demographics, video_info['product'], video_info['description'])
            ad_text = self.generator.generate_ad_text(messages)
            ad_queue.put(ad_text)
            self.output_results(video_info, ad_text)
            return ad_text
        except Exception as e:
            print(f"Error generating advertisement: {e}")

    def output_results(self, video_info, ad_text):
        """Display the results of the advertisement generation.

        Args:
            video_info (dict): Dictionary containing video details (file_name, product).
            ad_text (str): The generated advertisement text.
        """
        print(f"{video_info['product']}")
        print(f"{video_info['file_name']}")
        print(ad_text)

# Directly pass in the token when initializing the pipeline
# Replace 'your_huggingface_token_here' with your actual Hugging Face token
env_path = get_resource_path(".env")
load_dotenv(dotenv_path=env_path)
token = os.getenv("HF_TOKEN")
pipeline = AdvertisementPipeline(token=token)

if __name__ == "__main__":
    test_input = ('50+', 'Male', 'Indian', 'sad')
    pipeline.debug_mode = True
    pipeline.generate_advertisement(test_input)