# DeepSpeed Training Template

A modular and well-structured template for training language models using DeepSpeed with ZeRO optimization stages 2 and 3.

## 📁 Project Structure

```
deepspeed_template/
├── aws/
│   └── config.py                # S3 configuration utilities
├── deepspeed/
│   ├── zero-1.json              # ZeRO Stage 1 configuration
│   ├── zero-2.json              # ZeRO Stage 2 configuration
│   ├── zero-2-moe.json          # ZeRO Stage 2 for MoE models
│   └── zero-3.json              # ZeRO Stage 3 configuration
├── src/
│   ├── __init__.py              # Package initialization
│   ├── checkpoint.py            # Checkpoint management and S3 integration
│   ├── data.py                  # Data loading and tokenization utilities
│   ├── model.py                 # Model loading and custom MoE architecture
│   ├── train.py                 # Training, evaluation, and generation functions
│   └── utils.py                 # General utilities (seed setting, etc.)
├── test/
│   ├── __init__.py              # Test package initialization
│   ├── test_training_cpu.py    # CPU-only tests (no GPU required)
│   ├── test_training_gpu.py    # GPU training tests
│   └── test_checkpoint.py       # Checkpoint and S3 tests
├── assets/
│   └── images/                  # Training verification screenshots
├── config.yaml                  # Main configuration file (REQUIRED)
├── config.example.yaml          # Example configuration with documentation
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration (uv package manager)
├── uv.lock                      # Dependency lock file
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## ✅ Verified Training Results

This template has been tested and verified on AWS g4dn.12xlarge instances (4x Tesla T4 GPUs, 16GB each). Below are proofs of successful training:

### Training in Progress
![Training Progress](assets/images/Actually-training.png)
*DeepSpeed training running with ZeRO Stage 2, showing epoch progress and loss convergence*

### GPU Utilization
![GPU Utilization](assets/images/consuming-all-gpu-zero2.png)
*nvidia-smi output showing all 4 GPUs being utilized effectively with distributed memory allocation*

### Hardware Configuration

**Tested on**: AWS g4dn.12xlarge instance
- **GPUs**: 4x NVIDIA Tesla T4 (16GB each)
- **Total GPU Memory**: 64GB
- **vCPUs**: 48
- **RAM**: 192GB

## 🚀 Features

### Core Training
- **Modular Design**: Separate modules for training and configuration
- **ZeRO Stage 2**: Optimizer state partitioning with CPU offload
- **ZeRO Stage 3**: Full model parallelism (optimizer + parameters + gradients)
- **Mixed Precision Training**: FP16 for faster training and reduced memory
- **Multi-GPU Support**: Tested on 4x Tesla T4 GPUs (AWS g4dn.12xlarge)
- **Custom Model Support**: Built-in Qwen2 MoE architecture with 8 experts and gradient checkpointing

### Checkpointing & Resume
- **Periodic Checkpoints**: Automatic checkpoint saving every N steps
- **S3 Integration**: Non-blocking background upload to S3
- **Resume Support**: Resume training from local or S3 checkpoints
- **State Tracking**: Automatic tracking of epoch, step, loss, and optimizer state
- **Checkpoint Cleanup**: Automatic cleanup of old checkpoints

### Data & Monitoring
- **Progress Tracking**: Built-in progress bars and logging
- **Text Generation**: Test your model with custom prompts
- **Flexible Configuration**: Easy to switch between different DeepSpeed configurations
- **Reproducibility**: Configurable random seed for reproducible experiments
- **Data Loading**: Pre-built tokenization and data loading utilities
- **Comprehensive Testing**: CPU and GPU test suites for validation

## 📋 Requirements

### System Requirements

**CUDA Toolkit** is required for DeepSpeed to run. Make sure you have:
- NVIDIA GPU(s) with CUDA support
- CUDA Toolkit 11.8+ or 12.x installed
- Compatible NVIDIA drivers

To verify CUDA is available:
```bash
nvidia-smi
nvcc --version
```

### Python Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

#### Install uv (if not already installed)

```bash
pip install uv
```

#### Install dependencies

```bash
uv sync
```

Or install individually:

```bash
uv pip install torch>=2.0.0 torchvision>=0.15.0
uv pip install deepspeed>=0.12.0
uv pip install transformers>=4.35.0 datasets>=2.14.0
uv pip install tqdm>=4.65.0
```

> **Note**: The project includes `pyproject.toml` and `uv.lock` for dependency management. DeepSpeed requires CUDA toolkit to be installed on your system for GPU acceleration.

## 🎯 Quick Start

All configuration is done via the `config.yaml` file - no command line arguments needed!

### 0. Setup Configuration File

First, copy the example configuration:

```bash
cp config.example.yaml config.yaml
```

Or if `config.yaml` doesn't exist, it will be created with default values automatically.

### 1. Configure Your Training

Edit `config.yaml` to customize your training parameters:

```yaml
# config.yaml
data:
  dataset_name: "wikitext"
  batch_size: 8
  
training:
  num_epochs: 3
  
deepspeed:
  config_path: "deepspeed/zero-2-moe.json"
  
checkpoint:
  checkpoint_interval: 50
  
s3:
  enabled: false  # Set to true to enable S3 checkpointing
```

### 2. Run Training

**Multi-GPU Training (ZeRO Stage 2):**

```bash
# Uses all available GPUs
deepspeed main.py

# Or specify number of GPUs
deepspeed --num_gpus=4 main.py
```

**Single GPU Training (for testing):**

```bash
python main.py
```

**Custom Config File:**

```bash
# Use a different config file
deepspeed main.py --config config/my_custom_config.yaml
```

### Training with Custom Qwen2 MoE Model

The default configuration uses a custom 8-expert Mixture of Experts model.

**Model Specifications:**
- **Architecture**: Qwen2 with Mixture of Experts (MoE)
- **Parameters**: ~300M trainable parameters
- **Experts**: 8 experts, 2 active per token
- **Hidden Size**: 768
- **Layers**: 20
- **Attention**: Grouped-query attention (12 heads, 4 KV heads)
- **Features**: Gradient checkpointing enabled for memory efficiency

To train, simply configure `config.yaml` and run:

```bash
deepspeed --num_gpus=4 main.py
```

## 🧪 Running Tests

The project includes a comprehensive test suite to validate functionality.

### CPU-Only Tests (No GPU Required)

These tests validate core functionality without requiring CUDA/GPU:

```bash
# Run all CPU tests
pytest test/test_training_cpu.py -v

# Or run directly
python test/test_training_cpu.py
```

**What CPU tests validate:**
- ✓ Configuration file validity (ZeRO-2 and ZeRO-3)
- ✓ Module imports and dependencies
- ✓ Tokenizer loading and functionality
- ✓ Model loading
- ✓ CPU forward pass
- ✓ Utility functions (seed reproducibility)
- ✓ ZeRO configuration details

### Full Training Tests (GPU Required)

These tests require NVIDIA GPU with CUDA support:

```bash
# Run all GPU tests
pytest test/test_training.py -v

# Or run directly
python test/test_training.py
```

**What GPU tests validate:**
- ✓ DeepSpeed engine initialization (ZeRO Stage 2 & 3)
- ✓ Training loop execution
- ✓ Evaluation loop execution
- ✓ Forward/backward passes on GPU
- ✓ Checkpoint saving and loading
- ✓ Memory efficiency with ZeRO optimization
- ✓ Multi-GPU distributed training

> **⚠️ Important**: Tests in `test_training.py` will be automatically skipped if CUDA is not available. Make sure you have:
> - NVIDIA GPU(s) with CUDA support
> - CUDA Toolkit installed (11.8+ or 12.x)
> - Compatible NVIDIA drivers
> - PyTorch with CUDA support

### Run All Tests

```bash
# Run both CPU and GPU tests (GPU tests will skip if CUDA unavailable)
pytest test/ -v

# Run with detailed output
pytest test/ -v --tb=short

# Run specific test
pytest test/test_training.py::TestZeRoConfiguration::test_zero_stage2_config_exists -v
```

### Install Test Dependencies

Make sure pytest is installed:

```bash
uv add pytest
```

## ⚙️ Configuration Options

All configuration is done via `config.yaml`. Here's the complete structure:

### Data Configuration
```yaml
data:
  dataset_name: "wikitext"          # Dataset name from HuggingFace
  dataset_config: "wikitext-2-raw-v1"  # Dataset configuration
  batch_size: 8                      # Training batch size
  max_length: 128                    # Maximum sequence length
```

### Training Configuration
```yaml
training:
  num_epochs: 1                      # Number of training epochs
  max_train_steps: null              # Max steps per epoch (null for full epoch)
  max_eval_steps: null               # Max eval steps (null for full evaluation)
  log_interval: 10                   # Log every N steps
  seed: 42                           # Random seed for reproducibility
```

### DeepSpeed Configuration
```yaml
deepspeed:
  config_path: "deepspeed/zero-2-moe.json"  # DeepSpeed config file
  local_rank: -1                     # Local rank (set by DeepSpeed launcher)
```

### Model Configuration
```yaml
model:
  tokenizer_name: "Qwen/Qwen2.5-0.5B"  # Tokenizer from HuggingFace
  # model_name: "distilgpt2"           # Uncomment for different model
```

### Checkpoint Configuration
```yaml
checkpoint:
  output_dir: "./checkpoints"        # Checkpoint output directory
  save_checkpoint: false             # Save checkpoint after training
  checkpoint_interval: 50            # Save checkpoint every N steps
  keep_last_n_checkpoints: 3         # Number of local checkpoints to keep
  
  # Resume Configuration
  resume_from_checkpoint: null       # Set to checkpoint tag (e.g., "epoch0_step50")
  resume_step: null                  # Step number to resume from
```

### S3 Configuration
```yaml
s3:
  enabled: false                     # Enable S3 checkpoint upload/download
  bucket: null                       # S3 bucket name (required if enabled=true)
  prefix: "training/checkpoints"     # S3 folder path for checkpoints
  region: "us-east-1"                # AWS region for S3
  cleanup_after_upload: false        # Delete local checkpoints after S3 upload
```

### Generation Configuration
```yaml
generation:
  test_generation: true              # Test text generation after training
  generation_prompt: "The history of artificial intelligence begins with"
```

### Command Line Arguments

The script only accepts two optional command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration YAML file |
| `--local_rank` | `-1` | Local rank (automatically set by DeepSpeed launcher) |

## 💾 Checkpointing & Resume

### Automatic Periodic Checkpoints

Configure checkpoints in `config.yaml`:

```yaml
checkpoint:
  checkpoint_interval: 50          # Save every 50 steps
  output_dir: "./checkpoints"
  keep_last_n_checkpoints: 3
```

Then run:

```bash
deepspeed main.py
```

This will save checkpoints like:
- `epoch0_step50`
- `epoch0_step100`
- `epoch0_step150`
- `epoch0_end` (end of epoch)

### S3 Checkpointing

Enable automatic background upload of checkpoints to S3 in `config.yaml`:

```yaml
s3:
  enabled: true
  bucket: "my-training-bucket"
  prefix: "experiments/run-1"
  region: "us-east-1"
  
checkpoint:
  checkpoint_interval: 100
```

Then run:

```bash
deepspeed main.py
```

**Features:**
- ✅ Non-blocking background uploads (training continues while uploading)
- ✅ Automatic retry with exponential backoff
- ✅ Multi-node and multi-GPU support
- ✅ Progress tracking and detailed logging
- ✅ Automatic file distribution in multi-node scenarios

**S3 Structure:**
```
s3://my-training-bucket/experiments/run-1/
├── step_100/
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── ...
├── step_200/
└── final/
```

### Resume from Local Checkpoint

Configure resume in `config.yaml`:

```yaml
checkpoint:
  output_dir: "./checkpoints"
  resume_from_checkpoint: "epoch0_step100"
  resume_step: 100
```

Then run:

```bash
deepspeed main.py
```

The training will:
1. Load model and optimizer states
2. Restore epoch, step, and loss information
3. Continue from where it left off

### Resume from S3 Checkpoint

Configure S3 resume in `config.yaml`:

```yaml
s3:
  enabled: true
  bucket: "my-training-bucket"
  prefix: "experiments/run-1"
  region: "us-east-1"

checkpoint:
  resume_from_checkpoint: "epoch0_step100"
  resume_step: 100
```

Then run:

```bash
deepspeed main.py
```

The system will:
1. Check if checkpoint exists locally
2. Download from S3 if not found locally
3. Load checkpoint and restore training state
4. Continue training and upload new checkpoints to S3

### Advanced S3 Options

**Cleanup old local checkpoints:**

Configure in `config.yaml`:

```yaml
s3:
  enabled: true
  bucket: "my-bucket"
  cleanup_after_upload: true

checkpoint:
  keep_last_n_checkpoints: 2
```

This will:
- Keep only the last 2 checkpoints locally
- Delete older checkpoints after successful S3 upload
- Save local disk space during long training runs

**Environment Variables:**

You can also use environment variables for AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-west-2

deepspeed main.py
```

### Checkpoint Management

The `S3CheckpointManager` provides:

```python
from src.checkpoint import S3CheckpointManager
from aws.config import S3Config

# Initialize
config = S3Config(
    bucket_name="my-bucket",
    s3_prefix="training/run-1",
    local_checkpoint_dir="./checkpoints",
    keep_last_n_checkpoints=3,
)
checkpoint_mgr = S3CheckpointManager(config)

# Save checkpoint
checkpoint_mgr.save_checkpoint(model_engine, step=100)

# Load checkpoint
client_state = checkpoint_mgr.load_checkpoint(model_engine, step=100)

# Wait for uploads
checkpoint_mgr.wait_for_uploads()

# Cleanup old checkpoints
checkpoint_mgr.cleanup_old_checkpoints()
```

### DeepSpeed Configurations

#### ZeRO Stage 2 (`deepspeed/zero-2.json`)

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

#### ZeRO Stage 3 (`deepspeed/zero-3.json`)

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

### `src/model.py`

Model loading and initialization utilities:
- `get_model()`: Loads pretrained models from HuggingFace Hub
- `get_qwen2_moe_model()`: Creates custom Qwen2 MoE model from scratch with:
  - 8 experts with 2 active per token
  - Grouped-query attention for efficiency
  - Gradient checkpointing enabled
  - Router auxiliary loss for load balancing

### `src/data.py`

Data loading and tokenization utilities:
- `get_tokenizer()`: Loads and configures tokenizer from HuggingFace
- `get_dataloaders()`: Creates train/eval/test dataloaders with tokenization
- `tokenize_function()`: Tokenizes text examples for language modeling
- Supports HuggingFace datasets with automatic filtering and batching

### `src/train.py`

Training and inference logic:
- `train_epoch()`: Trains model for one epoch with progress tracking
- `evaluate()`: Evaluates model and computes loss and perplexity
- `generate_text()`: Generates text from prompts with configurable sampling
- `save_checkpoint()`: Saves model checkpoints using DeepSpeed
- `load_checkpoint()`: Loads model checkpoints

### `src/utils.py`

General utility functions:
- `set_seed()`: Sets random seeds across all libraries (Python, NumPy, PyTorch) for reproducibility
- Ensures deterministic behavior for consistent experimental results

### `main.py`

Main orchestration script that handles:
- Configuration loading from YAML file
- Random seed initialization for reproducibility
- Data loading with HuggingFace datasets
- Tokenizer initialization
- Model loading from HuggingFace Hub
- DeepSpeed engine initialization
- Training loop execution
- Validation and test set evaluation
- Text generation testing
- Checkpoint management

### `test/`

Comprehensive test suite for validation:

#### `test_training_cpu.py`
CPU-only tests that don't require GPU:
- Configuration file validation
- Module import checks
- Tokenizer and model loading
- CPU-based forward passes
- Utility function validation

#### `test_training.py`
Full training tests requiring GPU:
- DeepSpeed initialization (ZeRO Stage 2 & 3)
- Training and evaluation loops
- GPU forward/backward passes
- Checkpoint operations
- Memory efficiency validation
- Multi-GPU distributed training

> **Note**: GPU tests use `@pytest.mark.skipif(not torch.cuda.is_available())` to automatically skip when CUDA is unavailable.

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

### 1. Quick Test Run (Single GPU)

Configure `config.yaml`:

```yaml
training:
  max_train_steps: 50   # Limit training steps for quick test
  max_eval_steps: 20    # Limit eval steps
  num_epochs: 1

deepspeed:
  config_path: "deepspeed/zero-2.json"
```

Run:

```bash
python main.py
```

### 2. Training Qwen2 MoE from Scratch (Multi-GPU)

Configure `config.yaml`:

```yaml
data:
  batch_size: 8
  max_length: 1024

training:
  num_epochs: 5
  seed: 42

deepspeed:
  config_path: "deepspeed/zero-3.json"

checkpoint:
  save_checkpoint: true
  output_dir: "./checkpoints/qwen2_moe_run1"
  checkpoint_interval: 100

model:
  tokenizer_name: "Qwen/Qwen2.5-0.5B"
```

Run:

```bash
deepspeed --num_gpus=4 main.py
```

### 3. Full Training with Checkpointing (Multi-GPU)

Configure `config.yaml`:

```yaml
data:
  batch_size: 16

training:
  num_epochs: 5

deepspeed:
  config_path: "deepspeed/zero-3.json"

checkpoint:
  save_checkpoint: true
  output_dir: "./checkpoints/run1"
  checkpoint_interval: 50
```

Run:

```bash
deepspeed --num_gpus=4 main.py
```

### 4. Training with S3 Checkpointing (Cloud Training)

Configure `config.yaml`:

```yaml
data:
  batch_size: 16

training:
  num_epochs: 10

deepspeed:
  config_path: "deepspeed/zero-2.json"

checkpoint:
  checkpoint_interval: 100
  output_dir: "./checkpoints"

s3:
  enabled: true
  bucket: "my-ml-training"
  prefix: "experiments/gpt2-wikitext/run-001"
  region: "us-east-1"
```

Run:

```bash
deepspeed --num_gpus=4 main.py
```

### 5. Resume from S3 Checkpoint

Configure `config.yaml`:

```yaml
training:
  num_epochs: 10

deepspeed:
  config_path: "deepspeed/zero-2.json"

checkpoint:
  checkpoint_interval: 100
  resume_from_checkpoint: "epoch2_step500"
  resume_step: 500

s3:
  enabled: true
  bucket: "my-ml-training"
  prefix: "experiments/gpt2-wikitext/run-001"
```

Run:

```bash
deepspeed --num_gpus=4 main.py
```

### 6. Long Training with Checkpoint Cleanup

Configure `config.yaml`:

```yaml
data:
  batch_size: 8

training:
  num_epochs: 50

deepspeed:
  config_path: "deepspeed/zero-3.json"

checkpoint:
  checkpoint_interval: 50
  keep_last_n_checkpoints: 2

s3:
  enabled: true
  bucket: "my-ml-training"
  prefix: "long-training/qwen2-moe"
  cleanup_after_upload: true
```

Run:

```bash
deepspeed --num_gpus=4 main.py
```

### 7. Multi-Node Training with S3 (Advanced)

Configure `config.yaml`:

```yaml
data:
  batch_size: 32

training:
  num_epochs: 20

deepspeed:
  config_path: "deepspeed/zero-3.json"

checkpoint:
  checkpoint_interval: 200

s3:
  enabled: true
  bucket: "multi-node-training"
  prefix: "distributed-run/node-4x4gpu"
```

Run on all nodes:

```bash
deepspeed --num_nodes=4 --num_gpus=4 main.py
```

### 8. Using Multiple Config Files

You can maintain multiple config files for different experiments:

```bash
# Create custom configs
cp config.yaml config/experiment1.yaml
cp config.yaml config/experiment2.yaml

# Edit each config for different experiments
# Then run with specific config:
deepspeed main.py --config config/experiment1.yaml
deepspeed main.py --config config/experiment2.yaml
```

## 🐛 Troubleshooting

### Out of Memory Errors

Edit `config.yaml`:

1. Try Stage 3 instead of Stage 2:
   ```yaml
   deepspeed:
     config_path: "deepspeed/zero-3.json"
   ```

2. Reduce batch size:
   ```yaml
   data:
     batch_size: 4  # Reduce from 8
   ```

3. Reduce sequence length:
   ```yaml
   data:
     max_length: 64  # Reduce from 128
   ```

4. Enable gradient checkpointing in DeepSpeed config

### Slow Training

Edit `config.yaml`:

1. Try Stage 2 instead of Stage 3:
   ```yaml
   deepspeed:
     config_path: "deepspeed/zero-2.json"
   ```

2. Increase batch size if memory allows:
   ```yaml
   data:
     batch_size: 16  # Increase from 8
   ```

3. Disable CPU offloading in DeepSpeed config if you have enough GPU memory
4. Adjust `gradient_accumulation_steps` in DeepSpeed config

### Import Errors

```bash
# Make sure all dependencies are installed
uv sync
```

### Test Failures

If tests are failing:

**CPU Tests Failing:**
```bash
# Ensure all dependencies are installed
uv sync
uv pip install pytest

# Check Python version (requires Python 3.8+)
python --version
```

**GPU Tests Skipping/Failing:**
```bash
# Verify CUDA is available
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch CUDA installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"

# Reinstall PyTorch with CUDA support if needed
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: GPU tests are automatically skipped if CUDA is not available. This is expected behavior on CPU-only machines.

### S3 Upload/Download Issues

**Authentication Errors:**
```bash
# Verify AWS credentials are configured
aws configure list

# Or set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1

# Test S3 access
aws s3 ls s3://your-bucket-name/
```

**Bucket Permission Issues:**
```bash
# Check if bucket exists and you have access
aws s3 ls s3://your-bucket-name/

# Create bucket if needed
aws s3 mb s3://your-bucket-name --region us-east-1
```

**Slow S3 Uploads:**
- S3 uploads happen in the background and shouldn't slow training
- Check network bandwidth: `iftop` or `nethogs`
- Consider using larger `--checkpoint_interval` to reduce frequency
- Use `--cleanup_after_upload` to save disk space

**Resume Checkpoint Not Found:**
```bash
# List available checkpoints in S3
aws s3 ls s3://your-bucket/your-prefix/ --recursive

# Verify checkpoint tag name matches in config.yaml
checkpoint:
  resume_from_checkpoint: "epoch0_step100"  # Must match exact tag
```

**Multi-Node S3 Issues:**
- Ensure all nodes have AWS credentials
- Use IAM roles on EC2 instances (preferred over access keys)
- Check that all nodes can reach S3 endpoint
- Verify security groups allow outbound HTTPS (port 443)

## 📝 Customization

### Using Your Own Dataset

The template includes data loading utilities in `src/data.py`. To use your own dataset, modify the `get_dataloaders()` function:

```python
# src/data.py
def get_dataloaders(dataset_name, dataset_config, tokenizer, batch_size, max_length):
    """Create train/eval/test dataloaders."""
    
    # Option 1: Load from HuggingFace (already implemented)
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Option 2: Load from local files
    from datasets import Dataset
    import pandas as pd
    
    df = pd.read_csv("your_data.csv")
    dataset = Dataset.from_pandas(df)
    
    # Option 3: Load from custom source
    # Your custom data loading logic here
    
    # Then tokenize and create dataloaders
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length))
    # ... rest of the function
```

### Ensuring Reproducibility

To ensure reproducible experiments across runs, set the seed in `config.yaml`:

```yaml
training:
  seed: 42  # All runs with the same seed will produce identical results
```

For different experimental runs, use different seeds:

```yaml
training:
  seed: 123  # Different seed for different experiment
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

- Customize data loading in `src/data.py` for your specific datasets
- Add custom training strategies in `src/train.py`
- Create new DeepSpeed configurations in `deepspeed/`
- Add evaluation metrics and monitoring
- Implement additional model architectures
- Extend utilities in `src/utils.py`
- Add new test cases in `test/` directory

**When contributing:**
1. Run CPU tests to ensure basic functionality: `pytest test/test_training_cpu.py -v`
2. If you have GPU access, run full tests: `pytest test/test_training.py -v`
3. Add tests for new features you implement
4. Update documentation in this README

## 📄 License

This template is provided as-is for educational and research purposes.

---

**Happy Training! 🚀**
