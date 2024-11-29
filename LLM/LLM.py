from transformers import AutoTokenizer, AutoModelForCausalLM
# from model import *
# from yolov8 import *
# import data_store
import random

# Load the model and tokenizer globally
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = None
model = None

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

def simulate_thinking(thoughts, verbose=False):
    """Simulate the thought process for each thought item."""
    if verbose:
        for thought in thoughts:
            print(f"Thinking: {thought}")

def get_product_name(race, age_range, gender):
    """Select a product name based on the demographics."""
    try:
        product_list = advertisements[gender][age_range][race]
        if product_list:
            return random.choice(product_list)
        else:
            raise ValueError("No products found for the given demographics.")
    except KeyError:
        raise ValueError("Invalid demographic information provided.")


def generate_input_text_with_context(product_name, race, age_range, gender, tone, context, emotion):
    """Generate the input text for the prompt, including background context."""
    context_text = " ".join(context)
    if emotion.lower() == "sad":
        tone = "compassionate"
    
    return (
        f"Here are some background information: {context_text}\n\n"
        f"Create a compelling advertisement for our product, '{product_name}'. "
        f"Target Audience: {race} {gender} aged {age_range}, feeling {emotion}. "
        f"The advertisement should be in a {tone} tone, highlight unique features, "
        f"and create an emotional appeal. Include a catchy tagline. "
        f"Limit the response to 60 words, enclosed in quotes."
        f"Just print advertisement once, no more other information."
    )

def build_messages(input_text):
    """Build the messages list for the input text."""
    return [
        {"role": "system", "content": "You are an advertising expert. Your sole task is to provide advertisement content that perfectly meets my requirements, without any additional commentary."},
        {"role": "user", "content": input_text}
    ]

def generate_response(messages, max_new_tokens=120):
    """Generate a response for the input text using messages format."""
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

# Define a dictionary of product ads, grouped by gender, age, and race
advertisements = {
    "male": {
        "17-30": {
            "asian": [
                "icecream",  
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "suit",
                "glasses",
                "pants",
                "leathershoes",
                "sportscar",
                "GTA5",
                "switch",
                "massagegun",
                "wine",
                "albumenpowder",
                "treadmill",
                "essentials",
                
            ],
            "black": [
                "icecream",  
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "suit",
                "glasses",
                "pants",
                "leathershoes",
                "sportscar",
                "GTA5",
                "switch",
                "massagegun",
                "toilrtries", #
                "perfume", #
                "wine",
                "albumenpowder",
                "treadmill",
                "essentials",
            ],
            "white": [
                "icecream",
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "suit",
                "glasses",
                "pants",
                "leathershoes",
                "sportscar",
                "GTA5",
                "switch",
                "massagegun",
                "wine",
                "albumenpowder",
                "treadmill",
                "essentials",
            ],
            "indian": [
                "icecream",
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "suit",
                "glasses",
                "pants",
                "leathershoes",
                "sportscar",
                "GTA5",
                "switch",
                "massagegun",
                "wine",
                "albumenpowder",
                "treadmill",
                "essentials",
            ],
            "other": [
                "icecream",  
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "suit",
                "glasses",
                "pants",
                "leathershoes",
                "sportscar",
                "GTA5",
                "switch",
                "massagegun",
                "wine",
                "albumenpowder",
                "treadmill",
                "essentials",
            ],
        },
        "30-50": {
            "asian": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "suit",
                                "glasses",
                                "pants",
                                "leathershoes",
                                "sportscar",
                                "switch",
                                "massagegun",
                                "wine",
                                "treadmill",
                                "watch", #
            ],
            "black": [
                "icecream",  
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "suit",
                                "glasses",
                                "pants",
                                "leathershoes",
                                "sportscar",
                                "GTA5",
                                "switch",
                                "massagegun",
                                "toilrtries", #
                                "perfume", #
                                "wine",
                                "watch" #
                                "treadmill",
            ],
            "white": [
                "icecream",  
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "suit",
                                "glasses",
                                "pants",
                                "leathershoes",
                                "sportscar",
                                "GTA5",
                                "switch",
                                "massagegun",
                                "wine",
                                "treadmill",
                                "watch"
            ],
            "indian": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "sportscar",
                                                "GTA5",
                                                "switch",
                                                "massagegun",
                                                "wine",
                                                "treadmill",
                                                "watch"
            ],
            "other": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "sportscar",
                                                "GTA5",
                                                "switch",
                                                "massagegun",
                                                "toilrtries",
                                                "perfume",
                                                "wine",
                                                "treadmill",
                                                "watch" #
            ],
        },
        "50+": {
            "asian": [
                "icecream",  
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "suit",
                                "glasses",
                                "pants",
                                "leathershoes",
                                "shoes"
                                "massagegun",
                                "wine",
                                "curise",#
                                "massagechair",#
                                "watch",
                                "medicalequipment", #
                                "electronichealthbracelet", #
                                "healthcareproducts", #
            ],
            "black": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "shoes"
                                                "massagegun",
                                                "wine",
                                                "curise",#
                                                "massagechair",#
                                                "watch",
                                                "medicalequipment", #
                                                "electronichealthbracelet", #
                                                "toilrtries",#
                                                "perfume",#

            ],
            "white": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "shoes"
                                                "massagegun",
                                                "wine",
                                                "curise",#
                                                "massagechair",#
                                                "watch",
                                                "medicalequipment", #
                                                "electronichealthbracelet", #
            ],
            "indian": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "shoes"
                                                "massagegun",
                                                "wine",
                                                "curise",
                                                "massagechair",
                                                "watch",
                                                "medicalequipment",
                                                "electronichealthbracelet",
            ],
            "other": [
                "icecream",  
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "suit",
                                                "glasses",
                                                "pants",
                                                "leathershoes",
                                                "shoes"
                                                "massagegun",
                                                "wine",
                                                "curise",
                                                "massagechair",
                                                "watch",
                                                "medicalequipment",
                                                "electronichealthbracelet",
            ],
        },
    },
    "female": {
        "17-30": {
            "asian": [
                "icecream",
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "massagegun",
                "wine",
                "foundation",
                "lipstick",
                "bodycream",
                "popmart",#
                "bag",
                "dress",
                "eyeshadow",
                "instax",
                "starbucks",
                "lululemon",
                "heels",
                "essentials",
            ],
            "black": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "bag",
                                "dress",
                                "eyeshadow",
                                "instax",
                                "starbucks",
                                "lululemon",
                                "heels",
                                "essentials",
                                "travel",#
            ],
            "white": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "bag",
                                "dress",
                                "eyeshadow",
                                "instax",
                                "starbucks",
                                "lululemon",
                                "heels",
                                "essentials",
                                "skincareproducts" #
            ],
            "indian": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "bag",
                                "dress",
                                "eyeshadow",
                                "instax",
                                "starbucks",
                                "lululemon",
                                "heels",
                                "essentials",
            ],
            "other": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "bag",
                                "dress",
                                "eyeshadow",
                                "instax",
                                "starbucks",
                                "lululemon",
                                "heels",
                                "essentials",
            ],
        },
        "30-50": {
            "asian": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "bodycream",
                                "jewelry",#
                                "womanvitamin",#
                                "watches",#
                                "hermes",
                                "clhighheels",
                                "eyeshadow",
            ],
            "black": [
                "icecream",
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "massagegun",
                                                "wine",
                                                "foundation",
                                                "lipstick",
                                                "eyeshadow",
                                                "travel",#
                                                "jewelry",
                                                "womanvitamin",#
                                                "watches",#
                                                "hermes",
                                                "clhighheels",
            ],
            "white": [
                "icecream",
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "massagegun",
                                                "wine",
                                                "foundation",
                                                "lipstick",
                                                "jewelry",
                                                "womanvitamin",#
                                                "watches",
                                                "hermes",
                                                "clhighheels",
                                                "eyeshadow",
            ],
            "indian": [
                "icecream",
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "massagegun",
                                                "wine",
                                                "foundation",
                                                "lipstick",
                                                "jewelry",
                                                "womanvitamin",#
                                                "watches",
                                                "hermes",
                                                "clhighheels",
                                                "eyeshadow",
            ],
            "other": [
                "icecream",
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "massagegun",
                                                "wine",
                                                "foundation",
                                                "lipstick",
                                                "jewelry",
                                                "womanvitamin",#
                                                "watches",
                                                "hermes",
                                                "clhighheels",
                                                "eyeshadow",
            ],
        },
        "50+": {
            "asian": [
                "icecream",
                                                "KFC",
                                                "nikeshoes",
                                                "pepsi",
                                                "phone",
                                                "pumashoes",
                                                "soda",
                                                "sunglasses",
                                                "trolley case",
                                                "uniqlo",
                                                "applewatch",
                                                "massagegun",
                                                "wine",
                                                "foundation",
                                                "lipstick",
                                                "bodycream",#
                                                "jewelry",#
                                                "womanvitamin",#
                                                "watches",#
                                                "hermes",
                                                "clhighheels",
                                                "ikea",
                                                "organicvegetables",
                                                "laundrydetergent",
                                                "medicalequipment",
            ],
            "black": [
                "icecream",
                "KFC",
                "nikeshoes",
                "pepsi",
                "phone",
                "pumashoes",
                "soda",
                "sunglasses",
                "trolley case",
                "uniqlo",
                "applewatch",
                "massagegun",
                "wine",
                "foundation",
                "lipstick",
                "travel",#
                "jewelry",
                "womanvitamin",#
                "watches",#
                "hermes",
                "clhighheels",
                "ikea",
                "organicvegetables",
                "laundrydetergent",
                "medicalequipment",
            ],
            "white": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "jewelry",
                                "womanvitamin",#
                                "watches",#
                                "hermes",
                                "clhighheels",
                                "ikea",
                                "organicvegetables",
                                "laundrydetergent",
                                "medicalequipment",
            ],
            "indian": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "jewelry",
                                "womanvitamin",#
                                "watches",#
                                "hermes",
                                "clhighheels",
                                "ikea",
                                "organicvegetables",
                                "laundrydetergent",
                                "medicalequipment",
            ],
            "other": [
                "icecream",
                                "KFC",
                                "nikeshoes",
                                "pepsi",
                                "phone",
                                "pumashoes",
                                "soda",
                                "sunglasses",
                                "trolley case",
                                "uniqlo",
                                "applewatch",
                                "massagegun",
                                "wine",
                                "foundation",
                                "lipstick",
                                "jewelry",
                                "womanvitamin",#
                                "watches",#
                                "hermes",
                                "clhighheels",
                                "ikea",
                                "organicvegetables",
                                "laundrydetergent",
                                "medicalequipment",
            ],
        },
    },
}

BACKGROUND_INFO = [
    "Ice cream is a beloved frozen dessert made from dairy or plant-based milks with added sweeteners and flavorings to give it a smooth, creamy, rich taste. It is a versatile treat that comes in different flavors and can be enjoyed in cones, cups, or as part of creative desserts.",
    "KFC (Kentucky Fried Chicken) is a world-renowned fast food chain that serves a range of foods including hamburgers, French fries, and fried chicken.",
    "Nike athletic shoes are renowned for their comfort, durability and iconic design, and whether it's running, basketball, training, or lifestyle wear, Nike's extensive range ensures there's a perfect fit for everyone looking for high-quality, stylish and functional footwear.",
    "Pepsi is a globally recognized carbonated soft drink known for its bold, refreshing taste and the perfect combination of sweetness and fizz.",
    "The mobile phone is an essential communication device that has evolved into a powerful tool integrating advanced functions such as calling, text messaging, Internet browsing, photography and mobile applications.",
    "PUMA shoes are known for their blend of style, comfort and performance, making them a popular choice for athletes and casual wearers alike.",
    "Soda is a popular carbonated beverage known for its refreshing, bubbly taste and a variety of flavors, from classic cola and lemon-lime to fruity and exotic drinks.",
    "Sunglasses are a practical and fashionable accessory designed to protect the eyes from harmful UV rays, reduce glare and improve visual comfort and clarity in bright conditions.",
    "The Trolley Case is an essential travel companion designed for convenience and practicality. With its sturdy wheels, retractable handle and spacious compartments, it provides easy maneuverability and efficient packing for trips of any length.",
    "UNIQLO is a global apparel retailer offering a wide variety of apparel for men, women and children, including basics such as T-shirts, jeans and outerwear, as well as innovative products such as warm HEATTECH and breathable AIRism.",
    "Apple Watch is a versatile and sophisticated smartwatch that combines the power of advanced technology with stylish design. It provides users with notifications, fitness tracking, heart rate monitoring, electrocardiogram readings, GPS navigation and more.",
    "A timeless and essential piece of formal wear, the men's suit embodies sophistication, confidence and style. A well-fitted suit exudes confidence and professionalism and is appropriate for business meetings, formal events and special occasions.",
    "Men's eyewear is a fusion of function and fashion, designed to provide clear vision while complementing a variety of styles and lifestyles. Whether it's reading, computer use, or outdoor activities, there are glasses to meet specific needs.",
    "Men's trousers are an essential part of any wardrobe, designed to provide comfort, style and versatility. Key features typically include a tailored fit, utility pockets and high-quality fabrics, ensuring comfort and durability.",
    "Men's leather shoes are a timeless staple in a gentleman's wardrobe, offering a blend of elegance, sophistication and durability. Men's dress shoes can be paired seamlessly with formal attire for work, weddings, business meetings or upscale social events.",
    "A sports car is a high-performance luxury automobile designed for speed, agility and an exciting driving experience. A sports car is the epitome of automotive excellence, a statement of status and passion for driving.",
    "GTA5(Grand Theft Auto V) is an open-world action-adventure game developed by Rockstar Games that has received widespread acclaim for its immersive gameplay, stunning graphics, and rich storyline.",
    "The Nintendo Switch is a versatile gaming console that offers a unique hybrid design that allows users to play games on the go and at home. It makes it a great choice for entertainment between family games.",
    "A massage gun is a handheld, portable device designed to deliver targeted percussion therapy to muscles. This tool uses quick bursts of pressure to penetrate deep into muscle tissue, helping to relieve pain, improve circulation, and speed recovery after physical activity.",
    "Wine is a timeless and complex beverage made from fermented grapes, cherished for its rich flavors, aromatic notes and unique experience. Wine comes in many types, including red, white, rosé and sparkling, each with unique characteristics that suit different tastes and occasions.",
    "Egg white protein powder, also known as egg white protein powder, is a high-quality, complete protein supplement derived from egg whites. Rich in essential amino acids, it is an excellent source of protein for muscle growth, recovery, and overall health. Egg white powder is low in fat and carbohydrates, making it a great option for those looking to increase their protein intake without adding calories. It is commonly used in protein shakes, baking, and cooking, providing a versatile, easy-to-mix ingredient for fitness enthusiasts and individuals with dietary needs.",
    "A treadmill is a versatile piece of fitness equipment that allows users to walk, jog, or run in the comfort of their own home. Equipped with features such as adjustable speed, incline settings, and pre-set workout programs, treadmills are suitable for a variety of fitness levels and goals. Modern models often come with advanced features such as heart rate monitors, digital displays, and fitness app connectivity to enhance the workout experience. Whether it's for weight loss, cardiovascular health, or overall fitness, treadmills offer an effective, convenient way to stay active regardless of weather or time of day.",
    "Essentials are basic, high-quality pieces that form the foundation of a versatile wardrobe or everyday life. These pieces are designed for simplicity, comfort, and timeless style, making them perfect for every occasion. Whether it's a classic t-shirt, cozy loungewear, practical outerwear, or essential accessories, these essentials are carefully crafted to provide effortless style and lasting wear. Made from premium fabrics with attention to detail, these staples offer comfort and durability, blending functionality with understated elegance.",
    "A luxury men's watch is more than just a timepiece - it's a reflection of elegance, craftsmanship and status. High-end watches are carefully crafted through precision engineering and often contain complex functions such as chronographs, perpetual calendars and intricate mechanical movements.",
    "Toiletries include a wide range of cleaning products designed to keep the body, hair and face hygienic and well maintained.",
    "Men's perfume, or cologne, is more than just a scent - it's a reflection of personality, confidence and style. Black people seem to be more interested.",
    "Comfort sneakers for seniors are specially designed shoes that prioritize support, cushioning, and ease of wearing. These shoes typically feature lightweight construction, breathable materials, and non-slip soles to provide stability and prevent falls.",
    "Cruise vacations offer a unique blend of travel, relaxation, and adventure, all in one convenient package. Cruising allows travelers to explore multiple destinations without having to change hotels or transportation, as luxurious ships can ferry them from port to port.",
    "A massage chair is a luxurious piece of furniture designed to provide the relaxation and wellness benefits of a professional massage in the comfort of your own home. It can provide a full body experience targeting areas such as the neck, shoulders, back and legs to relieve muscle tension, improve blood circulation and reduce stress.",
    "Medical devices include a wide range of equipment and tools designed to assist in diagnosing, monitoring, treating and managing health conditions. From mobility aids such as wheelchairs and walkers to monitoring devices such as blood pressure monitors and blood glucose meters, medical devices play a vital role in improving quality of life and promoting independence. High-quality medical devices are designed to be reliable, safe and easy to use, ensuring individuals receive the care and support they need at home.",
    "Electronic health bracelets are wearable devices designed to monitor and track various health indicators to support an active, health-conscious lifestyle. These smart bracelets typically include features such as heart rate monitoring, step counting, sleep tracking, and activity reminders to help users understand their daily health. Electronic health bracelets are stylish and lightweight, comfortable to wear all day, and are an effective tool for staying health-conscious.",
    "Health care products include a wide range of items designed to support overall wellness, manage specific health conditions, and improve quality of life. These products include supplements, personal care items, medical devices, and health monitoring devices. Carefully crafted, many health care products incorporate scientifically backed ingredients or technologies to provide effective and reliable support.",
    "Foundation is an essential cosmetic product that provides a smooth, even base for makeup application. Foundations are now available to suit a wide range of skin tones and undertones, ensuring every user is perfectly matched and that a flawless, natural-looking makeup lasts all day.",
    "Lipstick is a classic, versatile makeup product that adds color, style and confidence to any look. Available in a variety of finishes including matte, satin, glossy and sheer, lipstick enhances the natural shape of lips while expressing personal style. From timeless reds and pinks to bold purples and deep wines, there is a perfect lipstick for every skin tone and occasion.",
    "Body cream is a nutrient-rich skin care product designed to moisturize and protect the skin, leaving it feeling soft and smooth. Body lotions contain emollients, humectants, and are often enriched with vitamins E and C to help lock in moisture and restore the skin's natural barrier.",
    "POP MART is a trendy and innovative brand known for its collectible art toys and blind box figures that have captured the hearts of collectors and enthusiasts. Each box contains a series of surprise figures, allowing you to excitedly discover new limited edition figures.",
    "For young women, a backpack is more than just a functional accessory; it combines style, convenience, and versatility. Designed to be both stylish and functional, these backpacks come in a variety of colors, patterns, and materials, from smooth leather to lightweight fabrics.",
    "Dresses for young women are a fashion staple that embodies elegance and versatility. With styles ranging from casual sundresses to chic cocktail dresses and sophisticated evening gowns, dresses are the perfect choice to express your personal style and embrace your femininity.",
    "Eyeshadow is a versatile makeup product that can enhance the eyes by adding depth, color, and dimension. Eyeshadow comes in a wide range of colors and finishes, from matte and shimmer to metallic and glitter, allowing for endless creativity in your makeup look.",
    "Instax cameras are a fun and instant way to capture and cherish memories. Compact, stylish and easy to use, Instax cameras are perfect for taking photos that are instantly printed, creating tangible keepsakes in seconds. With vibrant film options and a variety of shooting modes, users can personalize their photography experience and enjoy spontaneous, high-quality prints.",
    "Starbucks coffee is recognized around the world as a symbol of quality, consistency and rich flavor. With its diverse menu of expertly crafted beverages, Starbucks offers everything from classic brewed coffee and espresso to specialty beverages such as lattes, cappuccinos, and seasonal favorites.",
    "Lululemon is a premium brand known for its high-quality, fashionable and functional activewear designed to empower women in and out of the gym. Their products, including leggings, sports bras, tops and jackets, are crafted from advanced fabrics that provide exceptional comfort, flexibility and support.",
    "High heels are more than just shoes for young women – they are a symbol of confidence, elegance and personal style. Designed in a variety of heels, from classic stilettos to block and mid-heels, high heels add a touch of sophistication to any outfit. Crafted with comfort-enhancing features like cushioned insoles and supportive arches, modern high heels can be both stylish and functional.",
    "A beach vacation is the perfect combination of relaxation, adventure, and natural beauty. Whether you're soaking up the sun, playing in the waves, or strolling along the shoreline at sunset, a trip to the beach offers a chance to escape the hustle and bustle of everyday life. Many beach destinations offer a variety of activities, including swimming, snorkeling, surfing, beach volleyball, and the chance to take a boat trip and explore the local seafood delicacies.",
    "Skin care products are essential for maintaining healthy, radiant skin and addressing specific skin concerns. Made with a blend of nourishing ingredients, these products range from moisturizers and serums to cleansers and exfoliators, each playing a unique role in a comprehensive skin care routine.",
    "MAC lipstick is an iconic beauty product, renowned for its high-quality formula, rich pigmentation and wide range of shades. With finishes including matte, satin and gloss, and from classic red lips to subtle nudes or bold berries, MAC lipstick offers options for every look.",
    "Cartier jewelry represents the pinnacle of luxury, elegance and timeless craftsmanship. Cartier is renowned for its exquisite designs and unrivaled quality, from delicate necklaces and bracelets to iconic rings and earrings, often featuring precious metals and gemstones. Each item is meticulously crafted to embody contemporary style and classic tradition, making Cartier a symbol of sophistication and prestige.",
    "Women's vitamins are specially formulated to meet the unique nutritional needs of women at different stages of life. These supplements often include a comprehensive blend of essential vitamins and minerals, such as vitamins A, C, D, E, and vitamin B complex, as well as key nutrients like iron, calcium, and folate. Some formulas also contain additional ingredients, such as biotin for hair and nail health, antioxidants for glowing skin, and omega-3 fatty acids for heart and brain support. Women's vitamins are designed to boost energy, support immune health, and contribute to overall well-being, making them an important part of a healthy lifestyle.",
    "Cartier watches are the epitome of luxury, craftsmanship and timeless elegance. Renowned for their meticulous attention to detail and sophisticated design, Cartier timepieces combine high-quality materials such as gold, stainless steel and precious gemstones with precision Swiss watchmaking. From the iconic Tank and Santos de Cartier to the elegant Ballon Bleu collection, each watch embodies the Cartier tradition of innovation and style. Whether adorned with diamonds or kept simple, these watches are reliable timepieces and statements of prestige and sophistication.",
    "An Hermès handbag is more than just an accessory - it is a symbol of timeless elegance, exceptional craftsmanship and prestige. Each Hermès handbag is meticulously handcrafted by skilled artisans using the finest leathers and materials to create a piece that embodies luxury and sophistication. Iconic styles like the Birkin and Kelly have become synonymous with exclusivity and style, and are coveted by fashion connoisseurs and collectors around the world.",
    "Christian Louboutin (CL) heels are the epitome of luxury, elegance and a fashion statement. Known for their signature red soles, Louboutin's meticulously crafted shoes are made with great attention to detail and high-quality materials, offering the perfect blend of sophistication and bold style. These heels come in a variety of designs, from classic pumps to sophisticated, fashion-forward styles featuring unique embellishments, textures and silhouettes. Each pair embodies timeless beauty and craftsmanship, making them more than just footwear, but a piece of art that elevates any outfit.",
    "Cruise vacations offer a seamless blend of luxury, adventure, and relaxation while exploring beautiful destinations around the world. Cruise ships are equipped with first-class amenities, including fine dining, spas, pools, live entertainment, and a variety of activities tailored for guests of all ages. Travelers can visit multiple destinations without the hassle of packing and unpacking, as their accommodations travel with them. Cruise ships offer a unique travel experience that caters to both the adventure seekers and those looking for laid-back relaxation.",
    "IKEA is a globally recognised home furnishing brand, renowned for its modern, practical and affordable furniture and home accessories. IKEA offers a wide range of products to suit different tastes and lifestyles, from stylish sofas and dining tables to smart storage solutions and decorative items. Their innovative, space-saving designs are ideal for maximising small spaces, and the brand's commitment to sustainability is reflected in their many eco-friendly product lines.",
    "Organic vegetables are grown using natural farming methods without the use of synthetic pesticides, fertilizers, and GMOs. These vegetables are grown in nutrient-rich soil and cultivated using sustainable practices to ensure they are free of harmful chemicals, making them a healthier choice for consumers and better for the environment. Organic vegetables often have a fresher taste, higher nutritional content, and reduced exposure to synthetic residues, providing peace of mind for those who prioritize health and sustainability in their diet.",
    "Laundry detergent is an essential household product that keeps clothes clean, fresh, and free of dirt, stains, and odors. Whether for everyday laundry or tackling stubborn stains, a high-quality detergent helps preserve the life and appearance of your clothes.",
]

def get_relevant_background(product_name):
    """Get relevant background information based on the product name. Can be expanded to more complex logic."""
# Assuming a simple matching mechanism, select background information based on the product name
    if "ice cream" in product_name.lower():
        return [BACKGROUND_INFO[0]]
    elif "KFC" in product_name.lower():
        return [BACKGROUND_INFO[1]]
    elif "nikeshoes" in product_name.lower():
        return [BACKGROUND_INFO[2]]
    elif "Pepsi" in product_name.lower():
        return [BACKGROUND_INFO[3]]
    elif "phone" in product_name.lower():
        return [BACKGROUND_INFO[4]]
    elif "pumashoes" in product_name.lower():
        return [BACKGROUND_INFO[5]]
    elif "soda" in product_name.lower():
        return [BACKGROUND_INFO[6]]
    elif "sunglasses" in product_name.lower():
        return [BACKGROUND_INFO[7]]
    elif "trolley" in product_name.lower():
        return [BACKGROUND_INFO[8]]
    elif "uniqlo" in product_name.lower():
        return [BACKGROUND_INFO[9]]
    elif "applewatch" in product_name.lower():
        return [BACKGROUND_INFO[10]]
    elif "man-suit" in product_name.lower():
        return [BACKGROUND_INFO[11]]
    elif "man-glasses" in product_name.lower():
        return [BACKGROUND_INFO[12]]
    elif "man-pants" in product_name.lower():
        return [BACKGROUND_INFO[13]]
    elif "man-leathershoes" in product_name.lower():
        return [BACKGROUND_INFO[14]]
    elif "man-sportscar" in product_name.lower():
        return [BACKGROUND_INFO[15]]
    elif "man-GTA5" in product_name.lower():
        return [BACKGROUND_INFO[16]]
    elif "man-switch" in product_name.lower():
        return [BACKGROUND_INFO[17]]
    elif "massagegun" in product_name.lower():
        return [BACKGROUND_INFO[18]]
    elif "wine" in product_name.lower():
        return [BACKGROUND_INFO[19]]
    elif "man-albumenpowder" in product_name.lower():
        return [BACKGROUND_INFO[20]]
    elif "man-treadmill" in product_name.lower():
        return [BACKGROUND_INFO[21]]
    elif "essentials" in product_name.lower():
        return [BACKGROUND_INFO[22]]
    elif "watch" in product_name.lower():
        return [BACKGROUND_INFO[23]]
    elif "toilrtries" in product_name.lower():
        return [BACKGROUND_INFO[24]]
    elif "perfume" in product_name.lower():
        return [BACKGROUND_INFO[25]]
    elif "shoes" in product_name.lower():
        return [BACKGROUND_INFO[26]]
    elif "curise" in product_name.lower():
        return [BACKGROUND_INFO[27]]
    elif "massagechair" in product_name.lower():
        return [BACKGROUND_INFO[28]]
    elif "medicalequipment" in product_name.lower():
        return [BACKGROUND_INFO[29]]
    elif "electronichealthbracelet" in product_name.lower():
        return [BACKGROUND_INFO[30]]
    elif "healthcareproducts" in product_name.lower():
        return [BACKGROUND_INFO[31]]
    elif "foundation" in product_name.lower():
        return [BACKGROUND_INFO[32]]
    elif "lipstick" in product_name.lower():
        return [BACKGROUND_INFO[33]]
    elif "bodycream" in product_name.lower():
        return [BACKGROUND_INFO[31]]
    elif "popmart" in product_name.lower():
        return [BACKGROUND_INFO[34]]
    elif "bag" in product_name.lower():
        return [BACKGROUND_INFO[35]]
    elif "dress" in product_name.lower():
        return [BACKGROUND_INFO[36]]
    elif "eyeshadow" in product_name.lower():
        return [BACKGROUND_INFO[37]]
    elif "instax" in product_name.lower():
        return [BACKGROUND_INFO[38]]
    elif "starbucks" in product_name.lower():
        return [BACKGROUND_INFO[39]]
    elif "lululemon" in product_name.lower():
        return [BACKGROUND_INFO[40]]
    elif "heels" in product_name.lower():
        return [BACKGROUND_INFO[41]]
    elif "skincareproducts" in product_name.lower():
        return [BACKGROUND_INFO[42]]
    elif "mac" in product_name.lower():
        return [BACKGROUND_INFO[43]]
    elif "jewelry" in product_name.lower():
        return [BACKGROUND_INFO[44]]
    elif "womanvitamin" in product_name.lower():
        return [BACKGROUND_INFO[45]]
    elif "watches" in product_name.lower():
        return [BACKGROUND_INFO[46]]
    elif "hermes" in product_name.lower():
        return [BACKGROUND_INFO[47]]
    elif "clhighheels" in product_name.lower():
        return [BACKGROUND_INFO[48]]
    elif "travel" in product_name.lower():
        return [BACKGROUND_INFO[49]]
    elif "ikea" in product_name.lower():
        return [BACKGROUND_INFO[50]]
    elif "organicvegetables" in product_name.lower():
        return [BACKGROUND_INFO[51]]
    elif "laundrydetergent" in product_name.lower():
        return [BACKGROUND_INFO[52]]
    else:
        return []


def generate_input_text_with_context(product_name, race, age_range, gender, tone, context, emotion):
    """Generate the input text for the prompt, including background context."""
    context_text = " ".join(context)
    if emotion == "Sad":
        tone = "compassionate"
    
    return (
        f"Here are some background information: {context_text}\n\n"
        f"Create a compelling advertisement for our product, '{product_name}'. "
        f"Target Audience: {race} {gender} aged {age_range}, feeling {emotion}. "
        f"The advertisement should be in a {tone} tone, highlight unique features, "
        f"and create an emotional appeal. Include a catchy tagline. "
        f"Limit the response to 20-60 words, enclosed in quotes."
        f"Just print advertisement once, no more other information."
    )

def generate_ad_with_context(input_str, emotion, tone='Natural'):
    """Generate an ad using additional context."""
    thoughts = [
        "Considering the appealing factors for the target audience.",
        "Determining the most effective tone and style for this demographic.",
        "Ensuring the ad contains emotional triggers and a memorable tagline."
    ]
    simulate_thinking(thoughts)

    try:
        age_range, gender, race, emotion = input_str
        product_name = get_product_name(race.lower(), age_range, gender)
    except ValueError as e:
        print(e)
        return

    context = get_relevant_background(product_name)
    input_text = generate_input_text_with_context(product_name, race, age_range, gender, tone, context, emotion)
    messages = build_messages(input_text)

    response = generate_response(messages)
    ad_text = extract_ad_text(response)

    if ad_text:
        print("**Advertising Information:**")
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
    input_str = "17-30, male, Asian, happy"

    # Call the generator_llm_context
    generate_target_text(input_str)