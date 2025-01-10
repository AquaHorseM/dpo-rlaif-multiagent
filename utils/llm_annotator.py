import os
import json
import random
import re
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from api.utils import send_message_main
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class llmAnnotator:
    def __init__(self, dataset_path, dataset_name, model_name, label, output_path, prompt_type):
        """
        Initialize the llmAnnotator with the dataset path and LLM model.

        Args:
            dataset_path (str): Path to the offline dataset (e.g., data/Anthropic___hh-rlhf).
            model_path (str): Path to the offline LLM model directory.
        """

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_name = model_name
        output_path = os.path.join(output_path, label)
        if prompt_type != "strict":
            output_path = os.path.join(output_path, f"{model_name}_{prompt_type}")
        else:
            output_path = os.path.join(output_path, model_name)
        os.makedirs(output_path, exist_ok=True)
        file_name = f"{dataset_name}_annotated_dataset.json"
        output_path = os.path.join(output_path, file_name)
        self.output_path = output_path
        self.prompt_type = prompt_type
        # TODO: Only enable this line to overwrite the existing file
        with open(self.output_path, "w") as f:
            f.write("\n")
        with open(self.output_path.replace("annotated_dataset", "raw_dataset"), "w") as f:
            f.write("\n")

        if model_name == "llama7b": # working
            model_path = "local_models/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549"
        elif model_name == "mistral7b": # not working
            model_path = "local_models/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
            raise NotImplementedError("mistral7b not complete yet.")
        elif model_name == "deepseek7b":    # not working, why?
            model_path = "local_models/models--deepseek-ai--deepseek-llm-7b-base/snapshots/7683fea62db869066ddaff6a41d032262c490d4f"
        elif model_name == "llama32_1b":    # working
            model_path = "local_models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
        elif model_name == "llama2_13b":
            raise NotImplementedError("llama2_13b not available yet.")
        elif model_name == "yi6b":  # working
            model_path = "local_models/models--01-ai--Yi-6B/snapshots/80080be87ec5a0103f643195f2d9003b8068941b"
        elif model_name == "mixtral":
            raise NotImplementedError("mixtral not available yet.")
        elif model_name == "zephyr7b":
            raise NotImplementedError("zephyr7b not available yet.")
        elif model_name == "pythia28":  # working, but is too stupid to answer the question
            model_path = "local_models/models--EleutherAI--pythia-2.8b/snapshots/2a259cdd96a4beb1cdf467512e3904197345f6a9"
        elif model_name == "openai":
            print("openai model is only available online.")
            self.model = None
            self.tokenizer = None
            return
        elif model_name == "claude":
            print("claude model is only available online.")
            self.model = None
            self.tokenizer = None
            return
        elif model_name == "deepseek":
            print("deepseek model is only available online.")
            self.model = None
            self.tokenizer = None
            return
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        print(f"Loading LLM model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("LLM model loaded successfully.")

    def load_dataset(self, split):
        """
        Load the dataset split from the local path.

        Args:
            split (str): Dataset split to load (e.g., 'train', 'validation', 'test').

        Returns:
            dict: Processed dataset.
        """

        dataset_path = os.path.join(self.dataset_path, self.dataset_name)
        print(f"Loading dataset ({split} split) from {dataset_path}...")
        # dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=self.dataset_path, local_files_only=True)[split]
        dataset = load_from_disk(dataset_path)[split]

        if self.dataset_name == "Anthropic":
            data = defaultdict(lambda: {"responses": [], "pairs": []})
            for row in dataset:
                prompt = row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                try:
                    chosen_response = row["chosen"].split("\n\nAssistant:")[1]
                    rejected_response = row["rejected"].split("\n\nAssistant:")[1]
                except (KeyError, IndexError) as e:
                    print(f"Skipping row due to error: {e}, row: {row}")
                    continue
                if chosen_response == rejected_response:
                    continue
                data[prompt]["responses"].extend([chosen_response, rejected_response])

        elif self.dataset_name == "ListUltraFeedback":
            data = defaultdict(lambda: {"responses": [], "pairs": [], "scores": []})
            for row in dataset:
                prompt = row["prompt"]
                responses = row["all_responses"]
                data[prompt]["responses"].extend(responses)

        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        print(f"Loaded {len(data)} prompts.")
        return data
    
    def annotate_preferences(self, data, num_pairs_per_prompt, label="pair", test=False):
        if label == "pair":
            annotated_data = self.annotate_pair_preference(data, num_pairs_per_prompt, test=test)
        elif label == "random":
            annotated_data = self.annotate_random_preferences(data, num_pairs_per_prompt, test=test)
        else:
            raise ValueError(f"Unknown label: {label}")
        return annotated_data

    def annotate_pair_preference(self, data, num_pairs_per_prompt, test=False):
        """
        Annotate preferences twice for each pair of responses under each prompt.
        Change the order of the responses in each pair.

        Args:
            data (dict): Dataset containing prompts and response lists.

        Returns:
            dict: Dataset with annotated preference pairs.
        """
        annotated_data = defaultdict(lambda: {"pairs": []})
        if test:
            print("Test mode: Annotating preferences for at most 100 prompts.")
        for prompt, values in data.items():
            if len(annotated_data) == 100 and test:
                print("Test mode: Stopping early with 100 prompts.")
                break

            all_responses = values["responses"]
            annotated_data[prompt]["responses"] = all_responses
            for _ in range(num_pairs_per_prompt):
                if num_pairs_per_prompt > 1:
                    pair = random.sample(all_responses, 2)
                else:
                    pair = all_responses[:2]

                if self.prompt_type == "strict":
                    llm_input_1 = (
                        f"You are tasked with evaluating the quality of assistants' responses. I will provide you with a question and two responses.\n"
                        f"Your task is to choose which response is better.\n"
                        f"Question: {prompt}\n\nAssistant 1: {pair[0]}\n\nAssistant 2: {pair[1]}\n\n"
                        f"Which response is better? Answer strictly in the format of  'Assistant 1' or 'Assistant 2'. Do not add any explanation or extra text."
                    )
                    llm_input_2 = (
                        f"You are tasked with evaluating the quality of assistants' responses. I will provide you with a question and two responses.\n"
                        f"Your task is to choose which response is better.\n"
                        f"Question: {prompt}\n\nAssistant 1: {pair[1]}\n\nAssistant 2: {pair[0]}\n\n"
                        f"Which response is better? Answer strictly in the format of 'Assistant 1' or 'Assistant 2'. Do not add any explanation or extra text."
                    )
                elif self.prompt_type == "loose":
                    llm_input_1 = (
                        f"Suppose you are an annotator that makes mistakes on occasion. Here are a question and two responses.\n"
                        f"Your task is to choose which response is better.\n"
                        f"Question: {prompt}\n\nAssistant 1: {pair[1]}\n\nAssistant 2: {pair[0]}\n\n"
                        f"Which response will you (the annotator who makes mistakes occaionally) decide as the better one? Answer in the format of 'Assistant 1' or 'Assistant 2'. Do not add any explanation or extra text."
                    )
                    llm_input_2 = (
                        f"Suppose you are an annotator that makes mistakes on occasion. Here are a question and two responses.\n"
                        f"Your task is to choose which response is better.\n"
                        f"Question: {prompt}\n\nAssistant 1: {pair[0]}\n\nAssistant 2: {pair[1]}\n\n"
                        f"Which response will you (the annotator who makes mistakes occaionally) decide as the better one? Answer in the format of 'Assistant 1' or 'Assistant 2'. Do not add any explanation or extra text."
                    )
                elif self.prompt_type == "random":
                    raise NotImplementedError("Random prompt type not implemented yet.")
                
                if self.model is None:
                    preference_1 = send_message_main(self.model_name, llm_input_1)
                    preference_2 = send_message_main(self.model_name, llm_input_2)
                else:
                    inputs_1 = self.tokenizer(llm_input_1, return_tensors="pt", truncation=True, padding=True).to(device)
                    inputs_2 = self.tokenizer(llm_input_2, return_tensors="pt", truncation=True, padding=True).to(device)
                    outputs_1 = self.model.generate(**inputs_1, max_new_tokens=50)
                    outputs_2 = self.model.generate(**inputs_2, max_new_tokens=50)
                    preference_1 = self.tokenizer.decode(outputs_1[0], skip_special_tokens=True)
                    preference_2 = self.tokenizer.decode(outputs_2[0], skip_special_tokens=True)
                if test:
                    # print(f"\n\nLLM Input 1: {llm_input_1}\n\nLLM Input 2: {llm_input_2}\n\n")
                    print(f"\n\nPreference 1: {preference_1}\n\nPreference 2: {preference_2}\n\n")

                # Determine preference label
                preference_label_1 = extract_preference_from_output(llm_input_1, preference_1)
                preference_label_2 = extract_preference_from_output(llm_input_2, preference_2)
                if preference_label_1 == 0 and preference_label_2 == 1:
                    preference_label = 0
                elif preference_label_1 == 1 and preference_label_2 == 0:
                    preference_label = 1
                else:
                    preference_label = -1   # Ambiguous or inconsistant preference
                new_entry = {
                    "pair": pair,
                    "preference": preference_label,
                }
                annotated_data[prompt]["pairs"].append(new_entry)
                with open(self.output_path, "a") as f:
                    json.dump({"prompt": prompt, "result": [new_entry]}, f, indent=4)
                    f.write("\n")
                print("preferences:", preference_label_1, preference_label_2, preference_label)

                # store raw dataset without considering inconsistent preferences
                if preference_label_1 != -1:
                    with open(self.output_path.replace("annotated_dataset", "raw_dataset"), "a") as f:
                        new_entry = {
                            "pair": pair,
                            "preference": preference_label_1,
                        }
                        json.dump({"prompt": prompt, "result": [new_entry]}, f, indent=4)
                        f.write("\n")
                elif preference_label_2 != -1:
                    with open(self.output_path.replace("annotated_dataset", "raw_dataset"), "a") as f:
                        new_entry = {
                            "pair": pair,
                            "preference": 1 - preference_label_2,
                        }
                        json.dump({"prompt": prompt, "result": [new_entry]}, f, indent=4)
                        f.write("\n")
                else:
                    with open(self.output_path.replace("annotated_dataset", "raw_dataset"), "a") as f:
                        new_entry = {
                            "pair": pair,
                            "preference": -1,
                        }
                        json.dump({"prompt": prompt, "result": [new_entry]}, f, indent=4)
                        f.write("\n")

        return annotated_data

    
    def annotate_random_preferences(self, data, num_pairs_per_prompt=5, test=False):
        """
        Annotate preferences for randomly selected response pairs under each prompt.

        Args:
            data (dict): Dataset containing prompts and response lists.
            num_pairs_per_prompt (int): Number of pairs to sample per prompt.

        Returns:
            dict: Dataset with annotated preference pairs.
        """
        if test:
            print("Test mode: Annotating preferences for at most 100 prompts.")
        annotated_data = defaultdict(lambda: {"pairs": []})

        for prompt, values in data.items():
            if len(annotated_data) == 100 and test:
                print("Test mode: Stopping early with 100 prompts.")
                break

            responses = values["responses"]
            annotated_data[prompt]["responses"] = responses

            for _ in range(num_pairs_per_prompt):
                pair = random.sample(responses, 2)

                # Prepare the input for LLM preference annotation
                llm_input = (
                    f"You are tasked with evaluating the quality of assistants' responses. I will provide you with a question and two responses.\n"
                    f"Your task is to choose which response is better.\n"
                    f"Question: {prompt}\n\nAssistant 1: {responses[0]}\n\nAssistant 2: {responses[1]}\n\n"
                    f"Which response is better? Answer in the format of 'Assistant 1' or 'Assistant 2'. Do not add any explanation or extra text."
                )

                # Tokenize and generate preference signal
                if self.model is None:
                    preference = send_message_main(self.model_name, llm_input)
                else:
                    inputs = self.tokenizer(llm_input, return_tensors="pt", truncation=True, padding=True).to(device)
                    outputs = self.model.generate(**inputs, max_new_tokens=50)
                    preference = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Determine preference label
                preference_label = extract_preference_from_output(llm_input, preference)
                print("preferences:", preference_label)

                new_entry = {
                    "pair": pair,
                    "preference": preference_label,
                }
                annotated_data[prompt]["pairs"].append(new_entry)
                with open(self.output_path, "a") as f:
                    json.dump({prompt: [new_entry]}, f, indent=4)
                    f.write("\n")

        return annotated_data

    def save_annotated_dataset(self, annotated_data, output_path, label):
        """
        Save the annotated dataset to a JSON file.

        Args:
            annotated_data (dict): Annotated dataset with preference pairs.
            output_path (str): Path to save the JSON file.
        """
        output_path = os.path.join(output_path, label)
        output_path = os.path.join(output_path, self.model_name)
        file_name = "annotated_dataset.json"
        output_path = os.path.join(output_path, file_name)
        print(f"Saving annotated dataset to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(annotated_data, f, indent=4)
        print("Annotated dataset saved successfully.")

def extract_preference_from_output(input_text, output_text):
    if output_text.startswith(input_text):
        output_text = output_text[len(input_text):].strip()
    match = re.search(r"\b(Assistant 1|Assistant 2)\b", output_text)
    if match and match.group(1) == "Assistant 1":
        return 0
    elif match and match.group(1) == "Assistant 2":
        return 1
    else:
        # print(f"Could not extract preference from output text: {output_text}")
        # print(f"Input text: {input_text}")
        return -1   # If no valid preference is found

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate dataset using an offline LLM model.")
    parser.add_argument("--dataset_path", type=str, default="../data", help="Path to the offline dataset.")
    parser.add_argument("--dataset_name", type=str, default="ListUltraFeedback", help="Name of the offline dataset.", choices=["Anthropic", "ListUltraFeedback"])
    parser.add_argument("--model_name", type=str, required=True, help="Path to the offline LLM model.", choices=["llama7b", "mistral7b", "deepseek7b", "llama32_1b", "llama2_13b", "yi6b", "mixtral", "zephyr7b", "pythia28", "openai", "claude", "deepseek"])
    parser.add_argument("--output_path", type=str, default="../data/annotated", help="Path to save the annotated dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to annotate (default: train).")
    parser.add_argument("--num_pairs_per_prompt", type=int, default=5, help="Number of pairs to sample per prompt.")
    parser.add_argument("--test", type=bool, default=False, help="Test mode: Annotate preferences for at most 100 prompts.")
    parser.add_argument("--label", type=str, default="pair", help="Label for the annotated dataset.", choices=["pair", "random"])
    parser.add_argument("--prompt_type", type=str, default="strict", help="Prompt type for the LLM model to generate preferences.", choices=["strict", "loose", "random"])

    args = parser.parse_args()

    annotator = llmAnnotator(args.dataset_path, args.dataset_name, args.model_name, args.label, args.output_path, args.prompt_type)
    dataset = annotator.load_dataset(args.split)
    annotated_dataset = annotator.annotate_preferences(dataset, num_pairs_per_prompt=args.num_pairs_per_prompt, label=args.label, test=args.test)
    # annotator.save_annotated_dataset(annotated_dataset, args.output_path, label=args.label)
