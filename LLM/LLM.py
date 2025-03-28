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
        """Load the model and tokenizer using the provided token."""
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
        """Ensure required special tokens are configured"""
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if not self.tokenizer.eos_token:
            self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def build_system_prompt(self):
        """Generate system prompt template"""
        return (
            "You are a skilled copywriter at a global ad agency. Your only job is to deliver ad content matching my exact needs. "
            "Output must be in double quotes, strictly 20-30 words, no more, no less. "
            "If the text exceeds 30 words or falls below 20, adjust it before outputting. "
            "No extra text or comments allowed—only the ad content in quotes."
        )

    def build_user_prompt(self, demographics, product, context):
        """Construct user prompt from inputs"""
        tone = self.EMOTION_TONE_MAP.get(demographics['emotion'], "professional")
        context_text = " ".join(context)
        product_name = product

        return (
            f"Write a one-sentence creative ad text for {product_name}, strictly 20-30 words. "
            f"Target: {demographics['race']} {demographics['gender']}, aged {demographics['age_range']}, feeling {demographics['emotion']}. "
            f"Use a {tone} tone. Highlight unique features. Background: {context_text}. "
            f"Ensure 20-30 words, adjust if needed, and return only the ad content in double quotes."
        )

    def construct_messages(self, demographics, product, context):
        """Build complete message structure for LLM input"""
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(demographics, product, context)},
        ]

    def generate_ad_text(self, messages):
        """Generate a response for the input text using messages format."""
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
        """Process and clean model output"""
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return self.extract_ad_content(response)

    def extract_ad_content(self, text):
        """Extract content between quotes with improved regex"""
        match = re.search(r'"([^"]+)"', text)
        return match.group(1).strip() if match else text.strip()

class AdvertisementPipeline:
    """Orchestration class for ad generation pipeline"""

    def __init__(self, token):
        self.generator = AdvertisementGenerator(token)  # deliver token
        self.debug_mode = False

    def parse_demographics(self, input_data):
        """Validate and structure input data"""
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
        """Select appropriate video based on demographics"""
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
        """Output debug information if enabled"""
        if self.debug_mode:
            print(f"Selected Video: {video_info['file_name']}")
            print(f"Description: {video_info['description']}")
            print(f"Weight: {video_info['weight']}")
            print(f"Product: {video_info['product']}\n")

    def generate_advertisement(self, input_data):
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
        """Format and display results"""
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