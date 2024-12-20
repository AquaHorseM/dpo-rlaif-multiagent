import json
import os
from preference_datasets import get_batch_iterator  #, get_dataset
import transformers
import alpaca_eval
import argparse
from datasets import load_from_disk
from collections import defaultdict
import tqdm

def get_dataset(split: str, local_path: str = "./data/Anthropic___hh-rlhf") -> dict:
    """
    Load the Anthropic Helpful-Harmless dataset from a local directory and process it.
    
    Args:
        split (str): Dataset split to load ('train', 'validation', 'test').
        local_path (str): Path to the locally stored dataset directory.
        
    Returns:
        dict: A processed dataset with the following structure:
              {
                  'prompt1': {
                      'responses': List[str],
                      'pairs': List[Tuple[int, int]],
                      'sft_target': str
                  },
                  ...
              }
    """
    print(f"Loading Anthropic HH dataset ({split} split) from local path: {local_path}...")
    
    # Load the dataset from the local path
    dataset = load_from_disk(local_path)[split]
    print("Dataset loaded successfully.")

    def split_prompt_and_responses(example):
        """
        Extract prompt, chosen response, and rejected response from a dataset example.
        """
        prompt = example["chosen"][:example["chosen"].find("\n\nAssistant:") + len("\n\nAssistant:")]
        chosen_response = example["chosen"][len(prompt):]
        rejected_response = example["rejected"][len(prompt):]
        return prompt, chosen_response, rejected_response

    # Process the dataset into the required format
    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc=f"Processing {split} split"):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    print(f"Processed {len(data)} prompts from the {split} split.")
    return data

def get_ai_outputs(max_prompt_length=256, max_length=1024, data_fraction=1.0, num_turns=1, filter_out_fraction=0.,
                   model='chatgpt', cache_dir=args.cache_dir):
    tokenizer = transformers.AutoTokenizer.from_pretrained('huggyllama/llama-7b', cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if model == 'chatgpt':
        dataset_name = 'sharegpt'
    elif model == 'claude':
        dataset_name = 'shareclaude'
    elif model == 'gpt4':
        dataset_name = 'sharegpt4'
    prompt_iterator = get_batch_iterator([dataset_name,], tokenizer=tokenizer, split='train', batch_size=1, sft_mode=True,
                                         seed=0, n_epochs=1, cache_dir=cache_dir, shuffle=False,
                                         max_prompt_length=max_prompt_length, max_length=max_length, data_fraction=data_fraction, num_turns=num_turns,
                                         prefs_path=None, sampled_data_dir=None)

    chatgpt_instruction_truncoutput_pair = []
    instructions = []
    for batch in prompt_iterator:
        instruction = batch['prompt'][0]
        instruction = instruction[len('Human: '):-len('\n\nAssistant: ')]
        prompt_token_count = batch['prompt_input_ids'][0].shape[0]
        output = tokenizer.decode(batch['chosen_input_ids'][0][prompt_token_count:], skip_special_tokens=True)
        chatgpt_instruction_truncoutput_pair.append({'instruction': instruction,
                                                     'output': output})
        instructions.append(instruction)

    if filter_out_fraction > 0.:
        # alternate way that is more correct if there are eval sets or anything as such
        # prompt_iterator = get_batch_iterator(['sharegpt'], tokenizer=tokenizer, split='train', batch_size=1, sft_mode=True,
        #                                      seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
        #                                      max_prompt_length=max_prompt_length, max_length=max_length, data_fraction=filter_out_fraction, num_turns=num_turns,
        #                                      prefs_path=None, sampled_data_dir=None)
        # num_instructions_to_filter = len([batch for batch in prompt_iterator])
        # all_data = get_dataset(dataset_name, cache_dir=cache_dir, split='train', prefs_path=None, num_turns=num_turns, data_fraction=filter_out_fraction)
        all_data = get_dataset(split='train')
        num_instructions_to_filter = len(all_data)
        print(f'Filtering out {filter_out_fraction} of instructions, which is {num_instructions_to_filter} instructions')
        instructions = instructions[num_instructions_to_filter:]
        chatgpt_instruction_truncoutput_pair = chatgpt_instruction_truncoutput_pair[num_instructions_to_filter:]

    print(f'Number of instructions: {len(instructions)}')
    return chatgpt_instruction_truncoutput_pair, instructions

def process_llama_samples_from_dir(sample_folder, max_prompt_length=256, max_length=1024, num_turns=1, filter_out_fraction=0., model='chatgpt', cache_dir=args.cache_dir):
    sft_instruction_truncoutput_pair = []
    sft_instructions = []
    instructions_too_long = 0

    if filter_out_fraction > 0.:
        _, filtered_instructions = get_ai_outputs(max_prompt_length=max_prompt_length, max_length=max_length, data_fraction=1.0, num_turns=num_turns, filter_out_fraction=filter_out_fraction, model=model, cache_dir=cache_dir)

    for sample_file in os.listdir(sample_folder):
        sft_outputs = json.load(open(os.path.join(sample_folder, sample_file), 'r'))
        for instruction, sft_output in sft_outputs.items():
            instruction_trimmed = instruction[len('Human: '):-len('\n\nAssistant: ')]
            if instruction_trimmed in sft_instructions:
                continue
            # really long instructions exceed the context length for GPT-4
            if len(instruction_trimmed.split()) >= 2000 or len(instruction_trimmed) >= 8000:
                instructions_too_long += 1
                continue
            if filter_out_fraction > 0. and instruction_trimmed not in filtered_instructions:
                continue

            sft_instruction_truncoutput_pair.append({'instruction': instruction_trimmed,
                                                     'output': sft_output[0]})
            sft_instructions.append(instruction_trimmed)
    print(f'skipped {instructions_too_long} instructions that were too long')
    return sft_instruction_truncoutput_pair, sft_instructions

def match_instruction_outputs(instruct_out_1, instruction_set_1, instruct_out_2, instruction_set_2):
    matched_instruction_outputs = []
    for idx, instruction in enumerate(instruction_set_1):
        if instruction in instruction_set_2:
            matched_instruction_outputs.append({'instruction': instruction,
                                                'output_1': instruct_out_1[idx]['output'],
                                                'output_2': instruct_out_2[instruction_set_2.index(instruction)]['output']})
    return matched_instruction_outputs

def main(base_dir, model1_name, model2_name, max_length, data_fraction, max_num_comparisons, llm='claude', filter_out_fraction=0., out_folder_name=None):
    def _get_model_outputs_and_instructions(name):
        if name in ['chatgpt', 'gpt4', 'claude']:
            return get_ai_outputs(max_prompt_length=256, max_length=max_length, data_fraction=data_fraction, num_turns=1, filter_out_fraction=filter_out_fraction, model=name)
        else:
            return process_llama_samples_from_dir(os.path.join(base_dir, f'sharegpt2turn_noeos_maxlen{max_length}_{name}'),
                                                  max_prompt_length=256, max_length=max_length, num_turns=1, filter_out_fraction=filter_out_fraction,
                                                  model=model2_name if model2_name in ['claude', 'gpt4'] else 'chatgpt')

    out1, inst1 = _get_model_outputs_and_instructions(model1_name)
    out2, inst2 = _get_model_outputs_and_instructions(model2_name)
    matched_instruction_outputs = match_instruction_outputs(out1, inst1, out2, inst2)
    print(f'Number of matched instructions: {len(matched_instruction_outputs)}')

    if out_folder_name is None:
        comparison_name = f'{model1_name}_vs_{model2_name}'
    else:
        comparison_name = out_folder_name
    comparison_folder = os.path.join(base_dir, f'comparisons_{llm}', comparison_name)
    os.makedirs(comparison_folder, exist_ok=True)

    model1_outputs = []
    model2_outputs = []
    for idx, matched_instruction_output in enumerate(matched_instruction_outputs):
        model1_outputs.append({'instruction': matched_instruction_output['instruction'],
                               'output': matched_instruction_output['output_1'],
                               'generator': model1_name,})
        model2_outputs.append({'instruction': matched_instruction_output['instruction'],
                               'output': matched_instruction_output['output_2'],
                               'generator': model2_name,})
        if idx > max_num_comparisons:
            break

    model1_output_path = os.path.join(comparison_folder, model1_name + '.json')
    with open(model1_output_path, 'w') as f:
        json.dump(model1_outputs, f, indent=4)

    model2_output_path = os.path.join(comparison_folder, model2_name + '.json')
    with open(model2_output_path, 'w') as f:
        json.dump(model2_outputs, f, indent=4)

    if not isinstance(llm, list):
        alpaca_eval.evaluate(model_outputs=model1_output_path, annotators_config=llm, name=comparison_name, output_path=comparison_folder, reference_outputs=model2_output_path)
    else:
        for llm_single in llm:
            comparison_folder_single_llm = os.path.join(comparison_folder, str(llm_single))
            alpaca_eval.evaluate(model_outputs=model1_output_path, annotators_config=llm_single, name=comparison_name, output_path=comparison_folder_single_llm, reference_outputs=model2_output_path)

if __name__ == '__main__':
    def str_or_list(value):
        if value.startswith("[") and value.endswith("]"):
            # Remove brackets and split by commas to create a list
            return [v.strip() for v in value[1:-1].split(",")]
        return value  # Otherwise, treat it as a single string
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_name', type=str, default='temp1.0')
    parser.add_argument('--model2_name', type=str, default='chatgpt')
    parser.add_argument('--base_dir', type=str) # example: /home/ubuntu/.cache/rlaif/sharegpt2turn_llama7b_sft_maxlen1024_2023-09-11_21-52-36_584206/step-10000/
    parser.add_argument('--cache_dir', type=str, default='./local_models')
    parser.add_argument('--max_num_comparisons', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--num_turns', type=int, default=1)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--llm', type=str_or_list, default='claude')
    parser.add_argument('--filter_out_fraction', type=float, default=0.)
    parser.add_argument('--out_folder_name', type=str, default=None)
    args = parser.parse_args()

    main(base_dir=args.base_dir, model1_name=args.model1_name, model2_name=args.model2_name, max_length=args.max_length, data_fraction=args.data_fraction, max_num_comparisons=args.max_num_comparisons,
         llm=args.llm, filter_out_fraction=args.filter_out_fraction, out_folder_name=args.out_folder_name)