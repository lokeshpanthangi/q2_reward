# 🤖 AI Response Quality Reward Model

## 🌟 Project Overview

This project implements a comprehensive system for evaluating and scoring AI-generated responses based on quality and helpfulness. It consists of two specialized models working together:

1. **Text Generation Model** 🖋️ - Creates diverse responses to various prompts
2. **Reward Model** ⭐ - Evaluates and scores responses based on human preferences

The project demonstrates how to build an effective reward model that can be used for Reinforcement Learning from Human Feedback (RLHF) or as a standalone evaluation system for AI-generated content.

## 🔍 Why This Matters

As AI language models become more prevalent, the ability to automatically evaluate response quality becomes crucial. This project addresses that need by:

- Creating a dataset of ranked responses that captures human preferences
- Training a specialized model to predict these preferences
- Providing a framework for scoring and filtering AI-generated content

## 📁 Repository Structure

```
q2_reward/
├── answers.csv             # Dataset with prompts and ranked responses
├── reward_model/           # Trained reward model files (HuggingFace format)
│   ├── config.json         # Model configuration
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_args.json  # Training parameters
│   └── vocab.txt           # Tokenizer vocabulary
├── analyse.ipynb           # Jupyter notebook with training code
└── summary.md              # Technical documentation
```

> **Note:** The large model file (`model.safetensors`, 255MB) is hosted separately due to GitHub size limits. [Download it here](https://drive.google.com/file/d/1Xg_D4hO5idfQ8fmt1OIrHwYL0RgjFz9y/view?usp=sharing) and place it in the `reward_model/` directory.

## 🧠 How It Works

### 1. Data Collection Process 📊

The project follows a systematic approach to collect human preference data:

- **Prompt Creation**: We crafted 5 diverse prompts covering different content types:
  - Creative writing (AI story)
  - Educational explanation (quantum computing)
  - Humorous dialogue (cats)
  - Practical advice (time management)
  - Ethical analysis (AI in healthcare)

- **Response Generation**: For each prompt, we generated 4 different responses using the Microsoft Phi-2 model, varying parameters like temperature and seed to ensure diversity.

- **Human Evaluation**: Responses were manually ranked from 1 (best) to 4 (worst) based on quality, helpfulness, and relevance.

### 2. Dual-Model Architecture 🏗️

Our system uses two specialized models that work together:

#### Text Generation Model (Microsoft Phi-2)
- **Purpose**: Creates responses to prompts
- **Architecture**: Causal Language Model (AutoModelForCausalLM)
- **Function**: Takes a prompt as input and generates coherent text as output
- **Why**: Optimized for creative, diverse text generation

#### Reward Model (DistilBERT-based)
- **Purpose**: Evaluates response quality
- **Architecture**: Sequence Classification Model (DistilBertForSequenceClassification)
- **Function**: Takes a prompt-response pair and outputs a quality score
- **Why**: Efficient for scoring tasks, trained specifically on human preferences

### 3. Training Methodology 🔬

The reward model was trained using a preference learning approach:

1. **Data Preparation**: Converted rankings into preference pairs (better response vs. worse response)
2. **Model Initialization**: Started with DistilBERT and added a regression head
3. **Training Process**: Used the TRL (Transformer Reinforcement Learning) library with these parameters:
   - Learning rate: 5e-5
   - Batch size: 4
   - Training steps: 100
   - Optimizer: AdamW

### 4. Evaluation & Testing 📈

The trained model was evaluated on new responses to verify its ability to:
- Distinguish between high and low-quality responses
- Align with human preferences
- Generalize to similar but unseen prompts

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- Pandas
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/lokeshpanthangi/q2_rewards.git
cd q2_rewards

# Install required packages
pip install torch transformers trl pandas matplotlib datasets

# Download the model file
# Place the downloaded model.safetensors file in the reward_model/ directory
```

### Using the Reward Model

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

## 🔄 Training Your Own Model

To train your own reward model:

1. Run the Jupyter notebook `analyse.ipynb`
2. Follow the step-by-step process to:
   - Generate responses to prompts (or use your own)
   - Rank the responses (or use existing rankings)
   - Train the reward model
   - Evaluate the model

## 📊 Results & Findings

Our experiments showed that:

- The reward model successfully learned to distinguish between high and low-quality responses
- Scores correlated well with human rankings
- The model generalized reasonably well to new, unseen responses
- Response length, coherence, and relevance were key factors influencing scores

## 🔮 Future Work

Potential extensions to this project:

- Scale to more diverse prompts and domains
- Integrate with RLHF (Reinforcement Learning from Human Feedback) pipelines
- Explore multi-dimensional reward models for different aspects of quality
- Fine-tune the reward model on domain-specific data

## 🛠️ Technologies Used

- **Hugging Face Transformers**: For base language models
- **TRL (Transformer Reinforcement Learning)**: For reward model training
- **PyTorch**: As the underlying ML framework
- **Pandas**: For data manipulation and CSV handling
- **Matplotlib**: For visualization of reward scores

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Thanks to Hugging Face for their transformer models and TRL library
- Microsoft for the Phi-2 model used for response generation
