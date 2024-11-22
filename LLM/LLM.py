from transformers import AutoTokenizer, AutoModelForCausalLM
from model import *
from yolov8 import *
import data_store 

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

def simulate_thinking(thoughts, verbose=False):
    """Simulate the thought process for each thought item."""
    if verbose:
        for thought in thoughts:
            print(f"Thinking: {thought}")

def parse_input(input_str):
    """Parse input string into its components."""
    parts = [part.strip() for part in input_str.split(',')]
    if len(parts) < 4:
        raise ValueError("Input must contain product name, race, age range, and gender")
    return parts[0], parts[1], parts[2], parts[3]

def generate_input_text(product_name, race, age_range, gender, tone):
    """Generate the input text for the prompt."""
    return (
        f"Create a compelling advertisement for our product, '{product_name}'. "
        f"Target Audience: {race} {gender} aged {age_range}. "
        f"The advertisement should be in a {tone} tone, highlight unique features, "
        f"and create an emotional appeal. Include a catchy tagline. "
        f"Limit the response to 20-60 words, enclosed in quotes."
    )

def build_messages(input_text):
    """Build the messages list for the input text."""
    return [
        {"role": "system", "content": "You are an advertising expert. Your sole task is to provide advertisement content that perfectly meets my requirements, without any additional commentary."},
        {"role": "user", "content": input_text}
    ]

def generate_response(model, tokenizer, messages, max_new_tokens=60):
    """Generate a response for the input text using messages format."""
    # Flatten messages to create a single input text for the model
    input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_ad_text(response):
    """Extract ad text from the response."""
    start_idx = response.find('"') + 1
    end_idx = response.rfind('"')
    if start_idx > 0 and end_idx > start_idx:
        ad_text = response[start_idx:end_idx]
    else:
        ad_text = response.strip()
    return ad_text or "No valid advertisement text found."

def generate_ad_with_thinking(model, tokenizer, input_str, max_new_tokens=60, verbose=False):
    """Generate an ad with simulated thinking."""
    simulate_thinking(verbose=verbose)  # Simulate thought process with optional output
    parsed_info = parse_input(input_str)
    if parsed_info is None:
        return "Invalid input provided."
    product_name, race, age_range, gender = parsed_info
    input_text = generate_input_text(product_name, race, age_range, gender)
    response = generate_response(model, tokenizer, input_text, max_new_tokens)
    ad_text = extract_ad_text(response)
    return ad_text


# Main program
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tokenizer, model = load_model_and_tokenizer(model_name)
    except RuntimeError as e:
        print(e)
        exit()

    input_str = "KFC, Asian, 17-35, male"
    ad_text = generate_ad_with_thinking(model, tokenizer, input_str, tone='exciting', verbose=True)

    print("**Advertisement Message:**")
    print(ad_text)
