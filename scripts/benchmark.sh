# this script is only used to test the experiments

huggingface-cli login

set HF_HOME=E:\tsinghua\research\2024Fall\dpo-rlaif-multiagent\.cache\huggingface  #bash
setx HF_HOME "E:\tsinghua\research\2024Fall\dpo-rlaif-multiagent\.cache\huggingface"  # powershell
setx TRANSFORMERS_CACHE "E:\tsinghua\research\2024Fall\dpo-rlaif-multiagent\.cache\huggingface"
export PROJECT_CACHE=./.cache/rlaif

# wget
wget -P ./.cache/rlaif/sharegpt_data https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json 

# SFT
python train.py model=mistral7b batch_size=8 eval_batch_size=16 sample_during_eval=false loss=sft lr=1e-6 trainer=FSDPTrainer activation_checkpointing=True data_fraction=1.0 save_every=epoch_2 n_epochs=6 datasets=[sharegpt4]

# SFT stored in .cache/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b

# generate own preference data using your newly SFT-ed model
bash parallel_sample.sh ./.cache/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b/ mistral7b
# or
.\parallel_sample.ps1 "./.cache/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b/" "mistral7b" 1.0 "sharegpt"
# or
python generate_samples.py --archive "./.cache/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b/" --temperatures 1.0 --ff 6000 --data_fraction 1.0 --model_name llama7b --prompt_set sharegpt
# stored in: \.cache\huggingface\models--huggyllama--llama-7b

# Or download the pre-generated data
wget -P ./.cache/rlaif/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json
# or
Invoke-WebRequest -Uri "https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json" -OutFile ".\.cache\rlaif\sharegpt_data\comparisons_gpt4\mistral7bsft_vs_chatgpt\annotations.json"

# 
python label_ai_preferences.py --model1_name mistral7b_1.0 --base_dir ./.cache/huggingface/models--mistralai--Mistral-7B-v0.1 --max_num_comparisons 50000 --llm mistral7b

# DPO
python train.py loss=dpo loss.beta=0.05 model.archive=./.cache/rlaif/sharegpt_pythia28_2024-02-19_16-55-49_904051/epoch-3/policy.pt prefs_path=./.cache/rlaif/sharegpt_data/comparisons_gpt4/mistral7bsft_vs_chatgpt/annotations.json exp_name=pythia28 data_fraction=1.0 model=pythia28 save_every=epoch_1 n_epochs=3


# llm annotator
CUDA_VISIBLE_DEVICES=4 python ./utils/llm_annotator.py --model_path "local_models/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549"
# mistralai does not work, the model seem to have some error