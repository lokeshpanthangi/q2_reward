# Q2 Reward Model

A reward model for evaluating and scoring AI-generated responses based on quality and helpfulness.

## Overview

This project contains a trained reward model that can be used to score AI-generated responses. The model is based on DistilBERT architecture and has been fine-tuned on human preference data to evaluate response quality.

## Repository Contents

- `answers.csv`: Dataset containing prompts and ranked responses used for training
- `reward_model/`: Directory containing the trained reward model files (HuggingFace format)
- `analyse.ipynb`: Jupyter notebook with the model training and evaluation process
- `summary.md`: Detailed project documentation and technical information

## Quick Start

To use the reward model for scoring responses:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
tokenizer = AutoTokenizer.from_pretrained("./reward_model")

# Function to score a response
def score_response(prompt, response):
    inputs = tokenizer(prompt, response, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.item()
    return score

# Example usage
prompt = "Explain quantum computing"
response = "Quantum computers use qubits that can be both 0 and 1 at the same time."
score = score_response(prompt, response)
print(f"Response score: {score:.4f}")
```

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pandas (for data processing)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/q2_reward.git
cd q2_reward

# Install required packages
pip install torch transformers pandas
```

## Model Details

- Base model: DistilBERT
- Architecture: DistilBertForSequenceClassification
- Training parameters:
  - Learning rate: 5e-5
  - Batch size: 4
  - Training steps: 100
  - Optimizer: AdamW

## License

This project is available for research and educational purposes. See the LICENSE file for more details.

## Additional Information

For more detailed information about the model architecture, training process, and evaluation, please refer to the `summary.md` file.
