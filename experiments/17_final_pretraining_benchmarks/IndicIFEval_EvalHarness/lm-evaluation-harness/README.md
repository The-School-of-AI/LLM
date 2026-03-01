# Lm Eval Harness

In your python environment 
- Install lm eval harness from https://github.com/EleutherAI/lm-evaluation-harness


## Installation setup
```bash
# Create a new Conda environment with Python 3.10
conda create -n lmeval python=3.10
conda activate lmeval

# Clone the repository
git clone https://github.com/EleutherAI/lm-evaluation-harness

pip install -r requirements.txt

# Change directory
cd lm-evaluation-harness/

# Install the package in editable mode
pip install -e .
pip install -e .[vllm]
```

## Usage
```bash
model="path/to/model"
task_string="indicifeval_ground_xx,indicifeval_trans_xx" # Replace xx with language(s)
OUTPUT_PATH="results/$model/"

echo "########Evaluating model $model#########"
lm_eval --model vllm \
    --model_args "pretrained=$model,max_model_len=8192,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=True" \ # or enable_thinking=False
    --include_path "path/to/custom_configs/" \
    --gen_kwargs "temperature=0,max_gen_toks=4096" \
    --tasks $task_string \
    --batch_size auto \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --num_fewshot 0 \
    --apply_chat_template \
    --confirm_run_unsafe_code
```
