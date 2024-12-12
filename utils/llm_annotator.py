import os
import json
import random
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

class llmAnnotator:
    def __init__(self, dataset_path, model_path):
        """
        Initialize the llmAnnotator with the dataset path and LLM model.

        Args:
            dataset_path (str): Path to the offline dataset (e.g., data/Anthropic___hh-rlhf).
            model_path (str): Path to the offline LLM model directory.
        """
        self.dataset_path = dataset_path
        print(f"Loading LLM model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
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
        print(f"Loading dataset ({split} split) from {self.dataset_path}...")
        # dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=self.dataset_path, local_files_only=True)[split]
        dataset = load_from_disk("./data")[split]

        data = defaultdict(lambda: {"responses": [], "pairs": []})
        for row in dataset:
            prompt = row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
            chosen_response = row["chosen"].split("\n\nAssistant:")[1]
            rejected_response = row["rejected"].split("\n\nAssistant:")[1]

            data[prompt]["responses"].extend([chosen_response, rejected_response])

        print(f"Loaded {len(data)} prompts.")
        return data

    def annotate_pair_preference(self, data, test=False):
        """
        Annotate preferences twice for each pair of responses under each prompt.
        Change the order of the responses in each pair.

        Args:
            data (dict): Dataset containing prompts and response lists.

        Returns:
            dict: Dataset with annotated preference pairs.
        """
        annotated_data = defaultdict(lambda: {"pairs": []})
        for prompt, values in data.items():
            if len(annotated_data) == 100 and test:
                print("Test mode: Stopping early with 100 prompts.")
                break

            responses = values["responses"]
            annotated_data[prompt]["responses"] = responses

            llm_input_1 = f"\n\nHuman: {prompt}\n\nAssistant 1: {responses[0]}\n\nAssistant 2: {responses[1]}\n\nWhich response is better? Answer with 'Assistant 1' or 'Assistant 2'."
            llm_input_2 = f"\n\nHuman: {prompt}\n\nAssistant 1: {responses[1]}\n\nAssistant 2: {responses[0]}\n\nWhich response is better? Answer with 'Assistant 1' or 'Assistant 2'."
            inputs_1 = self.tokenizer(llm_input_1, return_tensors="pt", truncation=True, padding=True)
            inputs_2 = self.tokenizer(llm_input_2, return_tensors="pt", truncation=True, padding=True)
            outputs_1 = self.model.generate(**inputs_1, max_length=1000)
            outputs_2 = self.model.generate(**inputs_2, max_length=1000)
            preference_1 = self.tokenizer.decode(outputs_1[0], skip_special_tokens=True)
            preference_2 = self.tokenizer.decode(outputs_2[0], skip_special_tokens=True)

            # Determine preference label
            if "Assistant 1" in preference_1 and "Assistant 2" in preference_2:
                preference_label = 0
            elif "Assistant 2" in preference_1 and "Assistant 1" in preference_2:
                preference_label = 1
            else:
                preference_label = -1   # Ambiguous or inconsistant preference
            
            annotated_data[prompt]["pairs"].append({
                "pair": [responses[0], responses[1]],
                "preference": preference_label,
            })

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
                    f"\n\nHuman: {prompt}\n\nAssistant 1: {pair[0]}\n\nAssistant 2: {pair[1]}\n\n"
                    "Which response is better? Answer with 'Assistant 1' or 'Assistant 2'."
                )

                # Tokenize and generate preference signal
                inputs = self.tokenizer(llm_input, return_tensors="pt", truncation=True, padding=True)
                outputs = self.model.generate(**inputs, max_length=1000)
                preference = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Determine preference label
                if "Assistant 1" in preference:
                    preference_label = 0
                elif "Assistant 2" in preference:
                    preference_label = 1
                else:
                    preference_label = -1  # Ambiguous or no clear preference

                annotated_data[prompt]["pairs"].append({
                    "pair": pair,
                    "preference": preference_label,
                })

        return annotated_data

    def save_annotated_dataset(self, annotated_data, output_path, label):
        """
        Save the annotated dataset to a JSON file.

        Args:
            annotated_data (dict): Annotated dataset with preference pairs.
            output_path (str): Path to save the JSON file.
        """
        output_path = os.path.join(output_path, label)
        print(f"Saving annotated dataset to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(annotated_data, f, indent=4)
        print("Annotated dataset saved successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate dataset using an offline LLM model.")
    parser.add_argument("--dataset_path", type=str, default="./data", help="Path to the offline dataset.")   # debugging
    parser.add_argument("--model_path", type=str, required=True, help="Path to the offline LLM model.")
    parser.add_argument("--output_path", type=str, default="./data/labeled/", help="Path to save the annotated dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to annotate (default: train).")
    parser.add_argument("--num_pairs_per_prompt", type=int, default=5, help="Number of pairs to sample per prompt.")

    args = parser.parse_args()

    annotator = llmAnnotator(args.dataset_path, args.model_path)
    dataset = annotator.load_dataset(args.split)
    annotated_dataset = annotator.annotate_pair_preference(dataset)
    annotator.save_annotated_dataset(annotated_dataset, args.output_path, label="pair")
