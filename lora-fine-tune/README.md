# LoRA Fine Tune

## Setup
- Python: 3.11.13

### Install
```bash
pip install transformers peft datasets torch
```

## Download LLM
```bash
pip install -U "huggingface_hub[cli]"
```

### Llama3.2 3B
```bash
export HF_TOKEN=<your-huggingface-token>
export SAVE_PATH=hf_models/Llama-3.2-3B-Instruct
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```