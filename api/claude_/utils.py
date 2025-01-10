SLEEP_EVERYTIME = 2 # sleep for 1s every time, for what? I don't know!
TOKEN_LIMIT = 500  # token limit per message
TEMPERATURE = 1
MAX_RETRIES = 1
OPENAI_MODEL = "claude-3-5-sonnet-20241022" #for debugging, use the cheaper api
import openai
import time, yaml


def load_client_claude(key_path="claude_config.yaml"):
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

def send_message_claude(messages, client = None, agent_config = {}):
    '''
    A flexible function to send messages to openai
    Messages should be a tuple or list of tuples
    Each tuple should have two elements: role and content
    '''

    token_limit = agent_config.get("token_limit", TOKEN_LIMIT)
    temperature = agent_config.get("temperature", TEMPERATURE)
    max_retries = agent_config.get("max_retries", MAX_RETRIES)
    sleep_everytime = agent_config.get("sleep_everytime", SLEEP_EVERYTIME)
    model_name = agent_config.get("model", OPENAI_MODEL)
    context = get_context(messages)
    
    time.sleep(sleep_everytime)
        
    # connecting to Openai
    for i in range(max_retries):
        try:
            if client is None:
                client = load_client_claude()
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
            print(f"Error! Retry again.")
            time.sleep(5)
            continue
    return response.choices[0].message.content

if __name__=="__main__":
    client = load_client_claude("claude_config.yaml")
    prompt="Can you teach me quantum physics?"
    response = send_message_claude(prompt, client)
    print(response)