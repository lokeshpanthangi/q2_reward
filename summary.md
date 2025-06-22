# AI Response Quality Reward Model

## Project Overview
This project implements a reward model for evaluating and scoring AI-generated responses based on quality and helpfulness. The model is trained on human preference data and can be used to rank responses from language models.

## Directory Structure
```
q2_reward/
├── answers.csv             # Dataset containing prompts and ranked responses
├── reward_model/           # Trained reward model files (HuggingFace format)
│   ├── config.json         # Model configuration
│   ├── model.safetensors   # Model weights
│   ├── README.md           # Model description
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_args.json  # Training parameters
│   └── vocab.txt           # Tokenizer vocabulary
├── analyse.ipynb           # Jupyter notebook for model training and evaluation
└── summary.md              # This file - project summary
```

## Model Details
The reward model is based on DistilBERT architecture and was trained to score AI responses based on human preference data. It outputs a single score value representing the quality of a response given a prompt.

### Training Information
- Base model: DistilBERT
- Architecture: DistilBertForSequenceClassification
- Learning rate: 5e-5
- Batch size: 4
- Training steps: 100
- Optimizer: AdamW

## Dataset
The `answers.csv` file contains:
- Prompts covering various topics
- Multiple AI-generated responses for each prompt
- Human-assigned rankings for each response (1 = best, 4 = worst)

## Usage
The reward model can be used to:
1. Score and rank AI-generated responses
2. Filter low-quality content
3. Provide feedback on response quality
4. Guide reinforcement learning from human feedback (RLHF)

### Example Code
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

## Important Notes
- The reward model does NOT generate text - it only evaluates text quality
- Higher scores indicate better quality responses according to the training data
- The model was trained on a specific set of prompts and may perform differently on out-of-domain topics
