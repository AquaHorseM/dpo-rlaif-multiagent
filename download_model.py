import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, save_dir):
    """
    Download a Hugging Face model and tokenizer to a specified directory.

    Args:
        model_name (str): Name of the model to download from Hugging Face Hub.
        save_dir (str): Directory to save the model.
    """
    model_path = os.path.join(save_dir, model_name.replace('/', '--'))
    os.makedirs(model_path, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{model_path}'...")

    # Download and save the model
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path)

    # Download and save the tokenizer
    AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

    print(f"Model '{model_name}' successfully downloaded to '{model_path}'.")

if __name__ == "__main__":
    # Directory to save the models
    save_directory = "./local_models"

    # Models to download
    models_to_download = [
        "mistralai/Mistral-7B-v0.1",
        "alignment-handbook/zephyr-7b-sft-full",
    ]

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Download each model
    for model in models_to_download:
        download_model(model, save_directory)