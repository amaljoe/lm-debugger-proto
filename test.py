import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()
t: GPT2Model = model.transformer