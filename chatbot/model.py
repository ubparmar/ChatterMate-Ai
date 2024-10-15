# model.py

import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if using)
load_dotenv()

# Set up the Hugging Face API token
API_TOKEN = os.getenv("HF_API_TOKEN")
if not API_TOKEN:
    raise ValueError("Please set the HF_API_TOKEN environment variable.")

# Set up the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/{model_name}"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_response(user_input, conversation_history, model_name="facebook/blenderbot-400M-distill"):
    """
    Generates a response from the chatbot model using the Hugging Face Inference API.

    Parameters:
    - user_input (str): The user's input message.
    - conversation_history (list): A list of dictionaries containing the conversation history.
    - model_name (str): The name of the Hugging Face model to use.

    Returns:
    - assistant_reply (str): The assistant's reply to the user's input.
    """

    # Limit the conversation history to prevent exceeding the model's token limit
    max_history_length = 3  # Adjust this value as needed
    recent_history = conversation_history[-max_history_length:]

    # Build the conversation prompt
    inputs = []
    for message in recent_history:
        role = message['role']
        content = message['content']
        if role.lower() == 'user':
            inputs.append(f"Human: {content}")
        else:
            inputs.append(f"Assistant: {content}")
    inputs.append(f"Human: {user_input}")
    prompt = "\n".join(inputs)

    # Prepare the payload for the API request
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 128,  # Ensure this is within the model's token limit
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.03,
            # "truncation": "only_first"  # Uncomment to enable truncation if needed
        },
        "options": {
            "wait_for_model": True
        }
    }

    # Make the API request
    response = requests.post(
        API_URL.format(model_name=model_name),
        headers=headers,
        json=payload
    )

    # Check for errors
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Parse the response
    result = response.json()

    # Handle response format
    if isinstance(result, dict) and 'generated_text' in result:
        generated_text = result['generated_text']
    elif isinstance(result, list) and len(result) > 0:
        result = result[0]
        generated_text = result.get('generated_text', '')
    else:
        raise Exception(f"Unexpected API response format: {result}")

    # Extract the assistant's reply
    assistant_reply = generated_text.split('Assistant:')[-1].strip()

    return assistant_reply
