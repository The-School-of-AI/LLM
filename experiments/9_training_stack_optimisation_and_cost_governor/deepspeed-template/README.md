# DeepSpeed Training Template

A modular and well-structured template for training language models using DeepSpeed with ZeRO optimization stages 2 and 3.

## 📁 Project Structure

```
Template/
├── config/
│   └── deepspeed/
│       ├── zero-2.json          # ZeRO Stage 2 configuration
│       └── zero-3.json          # ZeRO Stage 3 configuration
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py        # Dataset and DataLoader utilities
│   └── train.py                 # Training, evaluation, and generation functions
├── main.py                      # Main entry point
└── README.md
```

## ✅ Verified Training Results

This template has been tested and verified on a 4x Tesla T4 GPU setup. Below are proofs of successful training:

### Training in Progress
![Training Progress](assets/images/Actually-training.png)
*DeepSpeed training running with ZeRO Stage 2, showing epoch progress and loss convergence*

### GPU Utilization
![GPU Utilization](assets/images/consuming-all-gpu-zero2.png)
*nvidia-smi output showing all 4 GPUs being utilized effectively with distributed memory allocation*

## 🚀 Features

- **Modular Design**: Separate modules for data loading, training, and configuration
- **ZeRO Stage 2**: Optimizer state partitioning with CPU offload
- **ZeRO Stage 3**: Full model parallelism (optimizer + parameters + gradients)
- **Mixed Precision Training**: FP16 for faster training and reduced memory
- **Flexible Configuration**: Easy to switch between different DeepSpeed configurations
- **Progress Tracking**: Built-in progress bars and logging
- **Text Generation**: Test your model with custom prompts
- **Checkpoint Support**: Save and load model checkpoints
- **AWS Ready**: Pre-configured scripts for g4dn.12xlarge (4x T4 GPUs)

## ☁️ Running on AWS

**Quick Start**: See [QUICK_START_AWS.md](QUICK_START_AWS.md) for copy-paste instructions!

**Detailed Guide**: See [aws_setup_g4dn.md](aws_setup_g4dn.md) for complete setup instructions.

```bash
# On AWS g4dn.12xlarge instance:
./run_on_g4dn.sh 1    # Quick test
./run_on_g4dn.sh 2    # Production training
```

## 📋 Requirements

```bash
pip install torch torchvision
pip install deepspeed
pip install transformers datasets
pip install tqdm
```

## 🎯 Quick Start

### Basic Training (ZeRO Stage 2)

```bash
python main.py --deepspeed_config config/deepspeed/zero-2.json
```

### Advanced Training (ZeRO Stage 3)

```bash
python main.py --deepspeed_config config/deepspeed/zero-3.json
```

### Custom Configuration

```bash
python main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --model_name distilgpt2 \
    --num_epochs 3 \
    --batch_size 16 \
    --max_length 256 \
    --save_checkpoint \
    --output_dir ./my_checkpoints
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `distilgpt2` | HuggingFace model name |
| `--dataset_name` | `wikitext` | Dataset name |
| `--dataset_config` | `wikitext-2-raw-v1` | Dataset configuration |
| `--batch_size` | `8` | Training batch size |
| `--max_length` | `128` | Maximum sequence length |
| `--num_epochs` | `1` | Number of training epochs |
| `--deepspeed_config` | `config/deepspeed/zero-2.json` | DeepSpeed config path |
| `--save_checkpoint` | `False` | Save checkpoint after training |
| `--output_dir` | `./checkpoints` | Checkpoint output directory |
| `--generation_prompt` | `The history of...` | Prompt for text generation |

### DeepSpeed Configurations

#### ZeRO Stage 2 (`config/deepspeed/zero-2.json`)

**What it does:**
- Partitions optimizer states across GPUs
- Offloads optimizer states to CPU memory
- Reduces memory footprint while maintaining speed

**Best for:**
- Medium-sized models (up to a few billion parameters)
- Multi-GPU setups with limited GPU memory
- Balancing speed and memory efficiency

**Key Features:**
- Optimizer state partitioning
- CPU offloading for optimizer states
- Gradient clipping
- Mixed precision (FP16)
- Learning rate warmup

#### ZeRO Stage 3 (`config/deepspeed/zero-3.json`)

**What it does:**
- Partitions optimizer states, gradients, AND model parameters
- Offloads both optimizer states and parameters to CPU
- Maximum memory savings for largest models

**Best for:**
- Very large models (billions to trillions of parameters)
- Limited GPU memory scenarios
- Training models that don't fit in GPU memory otherwise

**Key Features:**
- Full model parallelism
- Optimizer + parameter + gradient partitioning
- CPU offloading for optimizer and parameters
- Prefetching and communication overlap
- Model state gathering for checkpointing

## 🔧 Module Details

### `src/data/dataloader.py`

Handles all data loading and preprocessing:
- `get_tokenizer()`: Loads and configures tokenizer
- `preprocess_function()`: Tokenizes text data
- `WikiTextDataset`: Custom Dataset class
- `get_dataloaders()`: Creates train/eval/test dataloaders

### `src/train.py`

Contains all training logic:
- `train_epoch()`: Trains model for one epoch
- `evaluate()`: Evaluates model and computes perplexity
- `generate_text()`: Generates text from prompts
- `save_checkpoint()`: Saves model checkpoints
- `load_checkpoint()`: Loads model checkpoints

### `main.py`

Main orchestration script:
- Argument parsing
- Data loading
- Model initialization
- DeepSpeed initialization
- Training loop
- Evaluation and text generation

## 📊 Understanding ZeRO Stages

### Memory Distribution

```
ZeRO Stage 0 (Baseline):
GPU 0: [Model] [Optimizer] [Gradients]
GPU 1: [Model] [Optimizer] [Gradients]

ZeRO Stage 2:
GPU 0: [Model] [Optimizer Part 1] [Gradients Part 1]
GPU 1: [Model] [Optimizer Part 2] [Gradients Part 2]
       + CPU: [Optimizer States]

ZeRO Stage 3:
GPU 0: [Model Part 1] [Optimizer Part 1] [Gradients Part 1]
GPU 1: [Model Part 2] [Optimizer Part 2] [Gradients Part 2]
       + CPU: [Optimizer States] [Model Parameters]
```

### When to Use Each Stage

| Stage | Memory Savings | Speed | Use Case |
|-------|---------------|-------|----------|
| Stage 1 | 4x | Fastest | Small-medium models, plenty of GPU memory |
| Stage 2 | 8x | Fast | Medium models, limited GPU memory |
| Stage 3 | 15x+ | Moderate | Large models, very limited GPU memory |

## 🎓 Example Workflows

### 1. Quick Test Run

```bash
# Train for 1 epoch with limited steps
python main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --max_train_steps 50 \
    --max_eval_steps 20
```

### 2. Full Training with Checkpointing

```bash
# Train and save checkpoint
python main.py \
    --deepspeed_config config/deepspeed/zero-3.json \
    --num_epochs 5 \
    --batch_size 16 \
    --save_checkpoint \
    --output_dir ./checkpoints/run1
```

### 3. Custom Model Training

```bash
# Train a different model
python main.py \
    --model_name gpt2 \
    --deepspeed_config config/deepspeed/zero-3.json \
    --batch_size 4 \
    --max_length 512 \
    --num_epochs 3
```

## 🐛 Troubleshooting

### Out of Memory Errors

1. Try Stage 3 instead of Stage 2
2. Reduce `batch_size`
3. Reduce `max_length`
4. Enable gradient checkpointing (add to config)

### Slow Training

1. Try Stage 2 instead of Stage 3
2. Increase `batch_size` if memory allows
3. Disable CPU offloading if you have enough GPU memory
4. Adjust `gradient_accumulation_steps`

### Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Or install individually
pip install deepspeed transformers datasets torch tqdm
```

## 📝 Customization

### Using Your Own Dataset

Edit `src/data/dataloader.py` and modify the `get_dataloaders()` function:

```python
def get_dataloaders(
    dataset_name="your_dataset",
    dataset_config="your_config",
    # ... rest of arguments
):
    dataset = load_dataset(dataset_name, dataset_config)
    # Your custom preprocessing
```

### Adding Custom Metrics

Edit `src/train.py` and modify the `evaluate()` function:

```python
def evaluate(model_engine, data_loader, phase="Evaluation", max_steps=None):
    # ... existing code ...
    
    # Add your custom metrics
    accuracy = compute_accuracy(predictions, labels)
    print(f"Accuracy: {accuracy:.4f}")
```

## 📚 Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

Feel free to customize this template for your specific needs. Key areas to extend:

- Add more datasets in `src/data/`
- Implement custom training strategies in `src/train.py`
- Create new DeepSpeed configurations in `config/deepspeed/`
- Add evaluation metrics and monitoring

## 📄 License

This template is provided as-is for educational and research purposes.

---

**Happy Training! 🚀**
