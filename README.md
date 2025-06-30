# Clean Framework

A streamlined framework for training language models efficiently on various datasets.

## Overview

Clean Framework provides a structured approach to training language models on custom datasets. The framework handles data preprocessing, tokenization, training, and evaluation in a consistent way, allowing you to focus on designing your experiments rather than boilerplate code.

## Repository setup
```
# Clone the repository
git clone https://github.com/pe-hy/clean_framework.git
cd clean_framework

# Create a conda environment
conda create -n llms python=3.11

# Activate the environment
# On Windows, macOS, and Linux:
conda activate llms

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
clean_framework/
│
├── config/                     # Configuration files
│   ├── base.yaml               # Main configuration (modify for your project)
│   └── hf_config.py            # HuggingFace model configuration
│
├── data/                       # Data directory
│   ├── train.json              # Training data
│   ├── test.json               # Testing data
│   └── inference_data/         # Data for inference
│
├── data_generation/            # Scripts for generating data
│   └── generate_data.py        # Main data generation script
│
├── hpc_scripts/                # Scripts for HPC environments
│   ├── data_info.sh            # Get data statistics
│   ├── filter.sh               # Filter data
│   ├── gen_data.sh             # Generate data
│   ├── inference.sh            # Run inference
│   ├── job.sh                  # Training job
│   └── tokenizer.sh            # Create tokenizer
│
├── tokenizer/                  # Tokenizer files
│
├── utils/                      # Utility scripts
│   ├── create_tokenizer.py     # Tokenizer creation
│   ├── data.py                 # Data loading utilities
│   ├── evaluator.py            # Evaluation utilities
│   ├── filter_data.py          # Data filtering
│   ├── get_data_info.py        # Data statistics
│   └── inference.py            # Inference utilities
│
├── train.py                    # Main training script
├── requirements.txt            # Package list
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Data Format

Data must be in a JSON format that looks similar to this (input/output keys):

```json
[
  {
    "input": "E0 . T3 . T5 . T18",
    "output": "E45"
  },
  {
    "input": "E0 . T3 . T8 . T16 . T16",
    "output": "E1"
  },
  ...
]
```

## Data Tokenization

- Data will be tokenized like: `[PAD] [PAD] [BOS] E0 . T3 . T5 . T18 [OUT] E45 [EOS] ...` up to `model.block_size`
- The `[OUT]` delimiter token is added during preprocessing in `data.py`
- Anything after the delimiter, including the delimiter, is masked during training
- The model trains to predict the content following the delimiter token

## Configuration

Every time you work on a new project, you need to update the configuration:

- **config/base.yaml** contains default settings that should be modified for your specific project
- Pay special attention to model parameters (size, heads, layers) and project naming
- Run `python utils/get_data_info.py` to get information about your data length before setting `model.block_size`

## Step-by-Step Tutorial

### 1. Setup Environment

First, create a virtual environment and install the required dependencies:

```bash
# Create a conda environment
conda create -n llms python=3.11

# Activate the environment
# On Windows, macOS, and Linux:
conda activate llms

# Install dependencies
pip install -r requirements.txt

# In terminal, type:
wandb login
```

### 2. Generate Data

Create your dataset generation script in the `data_generation` directory:

```bash
# Place your script at data_generation/generate_data.py
python data_generation/generate_data.py
```

This should create `train.json` and `test.json` in the `data` directory.

### 3. Get Data Information

Analyze your dataset to determine appropriate model parameters:

```bash
python utils/get_data_info.py
```

Take note of the token statistics to set an appropriate `model.block_size` value in the configuration.

### 4. Configure Your Project

Edit `config/base.yaml` with your project-specific settings:

- Set `wandb.model_name` and `wandb.proj_name` with meaningful names
- Configure model size parameters (`n_layer`, `n_head`, `n_embd`)
- Set `model.block_size` based on your data statistics
- Adjust batch sizes and other training parameters as needed

### 5. Create Tokenizer

Generate a tokenizer based on your data:

```bash
python utils/create_tokenizer.py
```

This creates a tokenizer specific to your dataset's vocabulary.

### 6. Optional: Filter Data

If needed, filter examples that are too long:

```bash
python utils/filter_data.py
```

### 7. Train the Model

Start the training process:

```bash
python train.py
```

This will train the model according to your configuration and save checkpoints.

### 8. Run Inference

After training, run inference on new data:

```bash
python utils/inference.py
```

Place your inference data files in the `data/inference_data/` directory.

## Evaluation

The framework provides built-in evaluation metrics:

- Token-level accuracy
- Exact match accuracy
- Detailed evaluation examples logged to Weights & Biases


# Step-by-Step Tutorial for Clean Framework

This tutorial provides detailed instructions for setting up and using the Clean Framework for language model training.

## Prerequisites

- Python 3.11
- ROCm/CUDA-compatible GPU
- Git (for version control)

## 1. Project Configuration

The first step is to configure your project by editing `config/base.yaml`. Here are the key sections to update:

### Weights & Biases Configuration

```yaml
wandb:
  # Name your model
  model_name: "Pythia-12-8-256-MyProject" 
  # Your project name
  proj_name: "my-language-model-project"
  num_examples_reported: 100
```

### Model Configuration

```yaml
model:
  name: ${wandb.model_name}
  batch_size: 512
  accumulate_grad_batches: 1
  block_size: 1024  # Adjust based on your data
  epochs: 100

  n_layer: 12       # Number of transformer layers
  n_head: 8         # Number of attention heads
  n_embd: 256       # Embedding dimension
```

### Optimizer Configuration

```yaml
optim:
  lr_type: "linear"  # Options: "linear", "linear-reg", or None (cosine decay)
  lr: 2e-4           # Learning rate (used for cosine schedule)
```

## 2. Data Generation

Create a data generation script at `data_generation/generate_data.py` that produces data in the required JSON format:

```python
# Example of a simple data generation script (data_generation/generate_data.py)
import json
import os
import random

def generate_examples(num_examples):
    examples = []
    for _ in range(num_examples):
        # Generate your task-specific data here
        # Example for a simple addition task:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        input_text = f"Add {a} and {b}"
        output_text = f"{a + b}"
        
        examples.append({
            "input": input_text,
            "output": output_text
        })
    return examples

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate training data
    train_examples = generate_examples(10000)
    with open("data/train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test data
    test_examples = generate_examples(1000)
    with open("data/test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(test_examples)} test examples")

if __name__ == "__main__":
    main()
```

Run the data generation script:

```bash
python data_generation/generate_data.py
```

## 3. Analyze Data Characteristics

To properly set model parameters, analyze your data:

```bash
python utils/get_data_info.py
```

This will provide important statistics about your dataset:
- Average token count
- Maximum token count
- Token distribution
- Sample examples

Use this information to adjust your `model.block_size` in the configuration. The `block_size` should be large enough to accommodate most examples but not excessively large to waste memory.

## 4. Create a Tokenizer

Generate a tokenizer based on your data:

```bash
python utils/create_tokenizer.py
```

This creates a custom tokenizer optimized for your dataset's vocabulary.

## 5. Optional: Filter Data

If your dataset contains examples that are too long for your chosen `block_size`, you can filter them:

```bash
# Update the filter settings in config/base.yaml first:
data:
  filter:
    max_token_length: 1024  # Maximum token length for filtering examples

# Then run the filter script
python utils/filter_data.py
```

## 6. Start Training

Launch the training process:

```bash
python train.py
```

The training script will:
1. Load your data
2. Initialize the model
3. Train for the specified number of epochs
4. Save checkpoints based on validation accuracy
5. Log metrics to Weights & Biases (if configured)

## 7. Monitor Training

If you've configured Weights & Biases, you can monitor your training:

1. Open your web browser
2. Go to [https://wandb.ai](https://wandb.ai)
3. Navigate to your project
4. View training metrics and sample predictions

## 8. Run Inference

After training, you can run inference on new data:

1. Create JSON files in the `data/inference_data/` directory with the same format as your training data
2. Run the inference script:

```bash
python utils/inference.py
```

The script will generate predictions and calculate accuracy metrics.

## 9. Create Custom Tasks

To implement your own custom tasks:
1. Design your task and data format
2. Create a generation script in `data_generation/`
3. Update config as needed for your specific task
4. Follow the standard training workflow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
