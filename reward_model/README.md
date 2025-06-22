# Reward Model

This is a reward model for scoring AI responses based on quality and helpfulness.

## Training Details
- Learning rate: 5e-5
- Batch size: 4
- Training steps: 100
- Optimizer: AdamW

## Large Model File
The large model file `model.safetensors` (255MB) is not included in this repository due to GitHub file size limits. 

To obtain the model file:
1. Train the model using the `analyse.ipynb` notebook, or
2. Download it from [Hugging Face Hub](https://huggingface.co/) or another file sharing service where it might be hosted.

Once obtained, place the `model.safetensors` file in this directory to use the model for scoring responses.
