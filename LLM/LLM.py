from transformers import AutoTokenizer, AutoModelForCausalLM
from ad_pool.video_selection import *
import random
import re
from functools import lru_cache
from data_integration.data_interface import prediction_queue, ad_queue
import json
import time
import threading

# Load the model and tokenizer globally
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = None
model = None

class AdvertisementGenerator:
    """Class to encapsulate advertisement generation functionality"""
  
    EMOTION_TONE_MAP = {
        "sad": "empathetic and comforting",
        "angry": "assertive yet respectful",
        "happy": "energetic and enthusiastic",
        "neutral": "professional and trustworthy"
    }

    GENERATION_PARAMS = {
"max_new_tokens": 100, # The maximum number of tokens generated, controlling the length of the output text. Setting it to 120 means generating up to 120 tokens.
"min_new_tokens": 30, # The minimum number of tokens generated, ensuring that the output text is not too short. Setting it to 30 means generating at least 30 tokens.
"temperature": 0.4, # Controls the randomness of the generated text. Lower values ​​(such as 0.4) make the output more deterministic and conservative, and higher values ​​(such as 1.0) make the output more creative.
"top_p": 0.9, # Nucleus sampling parameter, controlling the range of tokens considered during generation. 0.9 means only considering tokens with cumulative probabilities in the top 90%, balancing diversity and quality.
"repetition_penalty": 1.2, # Parameter for penalizing duplicate tokens. Values ​​greater than 1.0 (such as 1.2) will reduce duplicate content, and values ​​less than 1.0 will increase duplicate content.
"do_sample": True, # Whether to use sampling methods to generate text. When set to True, use sampling (such as temperature and top_p) to generate more random text; when False, use greedy search to generate deterministic text.
"num_beams": 2, # The number of beams for beam search. The larger the value, the higher the quality of the generated text may be, but the computational cost is also higher. Setting it to 4 means using 4 beams for search.
"length_penalty": 0.5, # Controls the penalty factor for the length of the generated text. Values ​​greater than 1.0 encourage the generation of long text, and values ​​less than 1.0 encourage the generation of short text. 1.0 means no additional penalty is imposed.
"no_repeat_ngram_size": 3 # The size of n-grams that prohibit repetition. Setting it to 3 means prohibiting the generation of combinations containing repeated 3 tokens, reducing repetition.
}


    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model_and_tokenizer()
      
    """When calling repeatedly with the same parameters, the cached value is returned directly to avoid repeated calculations"""
    @lru_cache(maxsize=1)    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        global tokenizer, model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
          
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
        "Output must be in double quotes, 20-30 words, with no extra text or comments."
    )

    def build_user_prompt(self, demographics, product, context):
        """Construct user prompt from inputs"""
        tone = self.EMOTION_TONE_MAP.get(demographics['emotion'], "professional")
        context_text = " ".join(context)
        product_name = product

        return (
        f"Write a one-sentence creative ad text for {product_name} in 20-30 words. "
        f"Target: {demographics['race']} {demographics['gender']}, aged {demographics['age_range']}, feeling {demographics['emotion']}. "
        f"Use a {tone} tone. Highlight unique features. "
        f"Background: {context_text}. "
        "Return only the ad content in double quotes, nothing else."
    )

    def construct_messages(self, demographics, product, context):
        """Build complete message structure for LLM input"""
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(demographics, product, context)},
            #{"role": "assistant", "content": "You are a senior copywriter at a multinational advertising agency."}
        ] 

    def generate_ad_text(self, messages):
        """Generate a response for the input text using messages format."""
        #------------------
        # By using 'apply_chat_template':
        # 1. Automatically add bos_token at the beginning of the entire conversation and eos_token at the end
        # 2. apply_chat_template() will correctly insert delimiters based on the role of the message.
        # e.g. : [INST] and [/INST] are for user messages, <<SYS>> and <</SYS>> are for system messages.
        #------------------
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

        # print("【Original model output】:\n", response)
        return self.extract_ad_content(response)

    def extract_ad_content(self, text):
        """Extract content between quotes with improved regex"""
        match = re.search(r'"([^"]+)"', text)
        return match.group(1).strip() if match else text.strip()

class AdvertisementPipeline:
    """Orchestration class for ad generation pipeline"""
  
    def __init__(self):
        self.generator = AdvertisementGenerator()
        self.debug_mode = False

    def parse_demographics(self, input_data):
        """Validate and structure input data"""
        if len(input_data) != 4:
            raise ValueError("Invalid input format. Expected (age_range, gender, race, emotion)")
          
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
          
        selected = random.choice(videos)
        video_queue.put(selected[0])
        """    # 打印队列元素，不移除
        temp_list = []
        size = video_queue.qsize()
        for _ in range(size):
            item = video_queue.get()
            print(item)
            temp_list.append(item) #将元素放到临时列表
        for item in temp_list: #将临时列表中的元素全部放回队列
            video_queue.put(item)"""
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
            ad_queue.put(ad_text)  # 使用全局 ad_queue
            prediction_queue.put(("feedback"))
            self.output_results(video_info, ad_text)
            time.sleep(10)  # 等待反馈，可改为事件触发
            return ad_text
        except Exception as e:
            print(f"Error generating advertisement: {e}")

    def output_results(self, video_info, ad_text):
        """Format and display results"""
        print("**Advertising Information:**")
        print(f"{video_info['product']}: {video_info['description']}")
        print("\n**Personalized Advertising Message:**")
        print(ad_text)

# Initialize pipeline when module loads
pipeline = AdvertisementPipeline()

if __name__ == "__main__":
    # Example usage
    # Start Flask thread
    """flask_thread = threading.Thread(
        target=app.run,
        kwargs={'threaded': True, 'port': 5001}
    )
    flask_thread.daemon = True
    flask_thread.start()"""
    test_input = ('17-35', 'Male', 'Asian', 'happy')
    pipeline.debug_mode = True  # Enable for debugging
    pipeline.generate_advertisement(test_input)
