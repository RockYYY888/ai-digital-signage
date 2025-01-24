from transformers import AutoTokenizer, AutoModelForCausalLM
# from model import *
# from yolov8 import *
from video_selection import *
import random

# Load the model and tokenizer globally
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = None
model = None

# Create a emotion -> tone map
emotion_tone_map = {
    "sad": "empathetic and comforting",
    "angry": "assertive yet respectful",
    "happy": "energetic and enthusiastic",
    "neutral": "professional and trustworthy"
}


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            # Ensure pad_token_id and eos_token_id are distinct by using 'add_special_tokens' method.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        model.config.pad_token_id = tokenizer.pad_token_id
        
            # Ensure eos_token_id is set
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')
        model.config.eos_token_id = tokenizer.eos_token_id
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")


def generate_input_text_with_context(product,demographics, tone, context, emotion):
    """Generate the input text for the prompt, including background context."""
    tone = emotion_tone_map.get(emotion.lower(), "professional")
    context_text = " ".join(context)
    
    return (
        f"Your task is to produce a creative advertisement text strictly between 20-50 words. "
        f"Here is some background information: {context_text}\n\n"
        f"Create a compelling advertisement for our product, '{product}'. "
        f"Target Audience: {demographics['race']} {demographics['gender']} aged {demographics['age_range']}, feeling {emotion}. "
        f"The advertisement should be in a {tone} tone, highlighting unique features. "
        f"Provide the final advertisement content only, without any additional information. "
        f"Enclose the advertisement text in quotation marks."
    )


def build_messages(input_text):
    """Build the messages list for the input text."""
    #TODO: Try different content for system and user, although the output is good now, make it better 
    return [
        {"role": "system", "content": "You are an advertising expert. Your sole task is to provide advertisement content that perfectly meets my requirements, without any additional commentary."},
        {"role": "user", "content": input_text}
    ]

def generate_response(messages, max_new_tokens=120):
    """Generate a response for the input text using messages format."""
    #------------------
    # By using 'apply_chat_template':
    # 1. Automatically add bos_token at the beginning of the entire conversation and eos_token at the end
    # 2. apply_chat_template() will correctly insert delimiters based on the role of the message.
    # e.g. : [INST] and [/INST] are for user messages, <<SYS>> and <</SYS>> are for system messages.
    #------------------
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt") 

    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)

    # This is a decoded output: 
    # For skip_special_tokens=True, this will ensure that any special tokens are omitted from the generated output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_ad_text(response):
    """Extract ad text from the response."""
    #TODO: The current way of getting regular expressions can be a bit fragile, looking for ways to enhance it
    start_idx = response.find('"') + 1
    end_idx = response.rfind('"')
    if start_idx > 0 and end_idx > start_idx:
        ad_text = response[start_idx:end_idx]
    else:
        ad_text = response.strip()
    return ad_text or "No valid advertisement text found."


def generate_ad_with_context(input_data, emotion, tone='Natural'):
    """Generate an ad using additional context."""
    try:
        demographics = {
            'age_range': input_data[0],
            'gender': input_data[1],
            'race': input_data[2],
            'emotion': emotion
        }
    except ValueError as e:
        print(e)
        return
    
    targeted_videos = get_targeted_videos_with_ads(demographics["age_range"], demographics["gender"], demographics["race"])
    if targeted_videos: 

        random_video = random.choice(targeted_videos)
        # print(f"Randomly selected video: {random_video}")
 
        # access the video file name:
        random_video_file = random_video[0]
        # print(f"Randomly selected video file name: {random_video_file}") 

        # access the video description:
        random_video_description = random_video[1]
        # print(f"Randomly selected video description: {random_video_description}") 

        # access the video weight:
        random_video_weight = random_video[2]
        # print(f"Randomly selected video weight: {random_video_weight}") 

        # access the name of product:
        random_product_name = random_video[3]
        # print(f"Product name: {random_product_name}") 

        # comment all print statements to improve code readability, uncomment them when testing.
    else:
        print("No ads found.")
    
    input_text = generate_input_text_with_context(random_product_name,demographics, tone, random_video_description, emotion)
    messages = build_messages(input_text)

    response = generate_response(messages)
    ad_text = extract_ad_text(response)

    if ad_text:
        print("**Advertising Information:**")
        print(f"{random_product_name}:{random_video_description}")
        print("")
        print("**Personalized Advertising message:**")
        print(ad_text)
    else:
        print("Failed to generate ad text.")


def generate_target_text(input_str, emotion="happy", tone='Natural'):
    """Generate an advertisement based on the provided input."""
    generate_ad_with_context(input_str, emotion, tone)


# Load the model when the server starts
load_model_and_tokenizer()

if __name__ == "__main__":
    # Manual input for race, age range, gender, emotion
    input_str = ('17-35', 'Male', 'Asian', 'sad')  

    # Call the generator_llm_context
    generate_target_text(input_str)

