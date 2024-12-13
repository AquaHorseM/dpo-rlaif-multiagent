# CONFIG YOUR API HERE :
SLEEP_EVERYTIME = 2 # sleep for 1s every time, for what? I don't know!
TOKEN_LIMIT = 500  # token limit per message
TEMPERATURE = 1
MAX_RETRIES = 1
# MODEL = "gpt-4-turbo"
# MODEL = "gpt-4-turbo" #for debugging, use the cheaper api
MODEL = "gpt-4-turbo" #for debugging, use the cheaper api
from openai import OpenAI
import openai, os, time, yaml
import re

# this is for printing messages in terminal
DEBUG = False

def load_client(key_path="openai_config.yaml"):
    openai._reset_client()
    key = yaml.safe_load(open(key_path))
    for k, v in key.items():
        setattr(openai, k, v)
    return openai._load_client()

# make content in openai wanted formaty
def create_message(role, content):
    return {"role": role, "content": content}

def get_context(messages):
    if isinstance(messages, list) and isinstance(messages[0], dict):
        return messages
    context = []
    if isinstance(messages, tuple):
        messages = [messages]
    elif isinstance(messages, str):
        messages = [("user", messages)]
    for role, content in messages:
        context.append(create_message(role, content))
    return context

def send_message(messages, agent_config = {}, client = None):
    input_txt_path = client['input_txt_path']
    output_txt_path = client['output_txt_path']
    '''
    A flexible function to send messages to openai
    Messages should be a tuple or list of tuples
    Each tuple should have two elements: role and content
    '''
    token_limit = agent_config.get("token_limit", TOKEN_LIMIT)
    model_name = agent_config.get("model", MODEL)
    temperature = agent_config.get("temperature", TEMPERATURE)
    max_retries = agent_config.get("max_retries", MAX_RETRIES)
    sleep_everytime = agent_config.get("sleep_everytime", SLEEP_EVERYTIME)
    context = get_context(messages)
    
    time.sleep(sleep_everytime)
        
    # connecting to Openai
    for i in range(max_retries):
        try:
            if client is None:
                client = load_client()
            response = client.chat.completions.create(
                model=model_name, messages=context, temperature=temperature,
                max_tokens=token_limit, top_p=1
            )
            if response.choices[0].message.content is None:
                raise ValueError("Response is None!")
            break
        except Exception as e:
            if i == max_retries - 1:
                raise e
            time.sleep(5)
            continue
    return response.choices[0].message.content