import json

def convert_dataset(input_json_path):
    
    with open(input_json_path, 'r') as infile:
        content = infile.read()

    # Fix the format: Wrap objects in an array
    content = f"[{content}]".replace("}\n{", "},\n{")  # Add commas between objects
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    different_data = [
        entry for entry in raw_data
        if entry['result'][0]['pair'][0] != entry['result'][0]['pair'][1]
    ]
    print(f"Removed {len(raw_data) - len(different_data)} entries with identical answers out of {len(raw_data)} entries.")
    data = [
        entry for entry in different_data
        if any(result['preference'] in [0, 1] for result in entry['result'])
    ]
    removed_count = len(different_data) - len(data)
    removed_ratio = removed_count / len(different_data)
    print(f"Removed {removed_count} entries with only '-1' preferences out of {len(different_data)} entries.")
    print(f"Removed ratio: {removed_ratio}, consistency ratio: {1 - removed_ratio}")

    converted_data = []

    for entry in data:
        prompt = entry['prompt']
        pos_answers = []
        neg_answers = []
        
        result = entry['result'][0]
        if result['preference'] == 1:   # Assistant 1 is better
            pos_answers.append(result['pair'][0])
            neg_answers.append(result['pair'][1])
        elif result['preference'] == 0: # Assistant 2 is better
            pos_answers.append(result['pair'][1]) 
            neg_answers.append(result['pair'][0])

        converted_data.append({
            'prompt': prompt,
            'pos_answers': pos_answers,
            'neg_answers': neg_answers
        })

    # Write the converted data to a new JSON file
    output_json_path = input_json_path.replace('.json', '_formatted.json')
    with open(output_json_path, 'w') as outfile:
        json.dump(converted_data, outfile, indent=4)

if __name__=='__main__':
    input_path = '/home/zsxn/data/annotated/pair/claude/annotated_dataset.json'
    convert_dataset(input_path)