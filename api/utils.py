from api.openai_.utils import send_message_openai, load_client_openai
from api.claude_.utils import send_message_claude, load_client_claude
from api.deepseek_.utils import send_message_deepseek, load_client_deepseek
import multiprocessing
import traceback

def send_message_main(model_type, message, config_path = None, agent_config = {}):
    if model_type == "openai":
        if config_path is not None:
            client = load_client_openai(config_path)
        else:
            client = load_client_openai("api/openai_/openai_config.yaml")
        return send_message_openai(message, client, agent_config)
    elif model_type == "claude":
        if config_path is not None:
            client = load_client_claude(config_path)
        else:
            client = load_client_claude("api/claude_/claude_config.yaml")
        return send_message_claude(message, client, agent_config)
    elif model_type == "deepseek":
        if config_path is not None:
            client = load_client_deepseek(config_path)
        else:
            client = load_client_openai("api/deepseek_/deepseek_config.yaml")
        return send_message_deepseek(message, client, agent_config)

def send_message_wrapper(args):
    model_type, message, config_path, agent_config = args
    try:
        # Call the original send_message_main function and return the response
        response = send_message_main(model_type, message, config_path, agent_config)
        return response
    except Exception as e:
        # Handle exceptions by logging them
        print(f"Error occurred while processing message: {message}")
        print(str(e))
        print(traceback.format_exc())
        return f"Error occurred while processing message: {message}: {e}" # Return None in case of an error

def send_message_main_multiprocessing(model_type, messages, config_path=None, agent_config={}, process_count=8):
    # Prepare the arguments for each process
    args_list = [(model_type, message, config_path, agent_config) for message in messages]
    
    # Create a pool of processes to handle the messages in parallel
    with multiprocessing.Pool(processes=process_count) as pool:
        responses = pool.map(send_message_wrapper, args_list)

    # Return the list of responses
    return responses

# Example usage
if __name__ == "__main__":
    model_type = "claude"
    messages = ["I love you.", "I hate you.", "I like you.", "You are so annoying."]  # List of messages to send

    responses = send_message_main_multiprocessing(model_type, messages)
    print(responses)
