#!/bin/bash
export PYTHONPATH="/home/zsxn/dpo-rlaif-multiagent"

# local model

model_name_list=("llama7b" "yi6b" "llama32_1b" "pythia28")
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Loop through model names
for i in "${!model_name_list[@]}"; do
    model_name="${model_name_list[$i]}"
    gpu_id=$((i % num_gpus)) # Assign GPU in a round-robin fashion
    echo "Running model $model_name on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python ./utils/llm_annotator.py --model_name "$model_name" --label "pair" &
done
wait
echo "All models processed!"


# all models
# model_name_list=("llama7b" "yi6b" "llama32_1b" "pythia28")    # local models
CUDA_VISIBLE_DEVICES=0 python ./utils/llm_annotator.py --model_name "llama7b" --label "pair" --num_pairs_per_prompt 1 --dataset_name "Anthropic"
python ./utils/llm_annotator.py --model_name "yi6b" --label "pair" --num_pairs_per_prompt 1
python ./utils/llm_annotator.py --model_name "llama32_1b" --label "pair" --num_pairs_per_prompt 1
python ./utils/llm_annotator.py --model_name "pythia28" --label "pair" --num_pairs_per_prompt 1

# model_name_list=("openai" "claude" "deepseek")    # api models
python ./utils/llm_annotator.py --model_name "openai" --label "pair" --num_pairs_per_prompt 1 --dataset_name "Anthropic" --prompt_type "loose"
python ./utils/llm_annotator.py --model_name "claude" --label "pair" --num_pairs_per_prompt 1
python ./utils/llm_annotator.py --model_name "deepseek" --label "pair" --num_pairs_per_prompt 1