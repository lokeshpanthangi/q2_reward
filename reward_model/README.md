# 🌟 Reward Model

This is a reward model for scoring AI responses based on quality and helpfulness.

## 📊 Training Details
- Learning rate: 5e-5
- Batch size: 4
- Training steps: 100
- Optimizer: AdamW

## 📥 Large Model File
The large model file `model.safetensors` (255MB) is not included in this repository due to GitHub file size limits. 

### ⬇️ Download Link
**[Download model.safetensors here](https://drive.google.com/file/d/1Xg_D4hO5idfQ8fmt1OIrHwYL0RgjFz9y/view?usp=sharing)**

After downloading, place the file in this directory (`reward_model/`) to use the model for scoring responses.

## 🧠 Model Architecture
- Base model: DistilBERT
- Architecture: DistilBertForSequenceClassification
- Output: Single score value representing response quality
- Input format: Prompt and response pair

## 📋 Usage Instructions

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

## 🔍 Score Interpretation
- Higher scores indicate better quality responses
- Scores typically range from -3 to +3
- The model was trained to prefer:
  - Relevance to the prompt
  - Factual accuracy
  - Coherence and clarity
  - Helpful and informative content
