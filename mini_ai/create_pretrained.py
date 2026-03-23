"""
Quick pre-training script for Mini AI
Creates pre-trained weights efficiently
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MiniAI


def get_training_text():
    """Get training text data"""
    return """
The quick brown fox jumps over the lazy dog. 
Artificial intelligence is transforming the world.
Machine learning enables computers to learn from data.
Deep neural networks can recognize complex patterns.
Natural language processing helps machines understand text.
Computer vision allows machines to see and interpret images.
Reinforcement learning trains agents through trial and error.
Transfer learning applies knowledge from one task to another.
Attention mechanisms help models focus on relevant information.
Transformers have revolutionized natural language processing.
Pre-training on large datasets enables few-shot learning.
The model learns representations of words and concepts.
Gradient descent optimizes the loss function iteratively.
Backpropagation computes gradients through the network.
Regularization prevents overfitting to training data.
Data augmentation increases the effective training set size.
Batch normalization stabilizes training dynamics.
Dropout randomly deactivates neurons during training.
The loss function measures the difference between predictions and targets.
Cross entropy is commonly used for classification tasks.
Mean squared error is used for regression problems.
Optimization algorithms like Adam adapt learning rates.
Learning rate scheduling can improve convergence.
Early stopping prevents excessive training.
Model validation ensures generalization to unseen data.
The training loop iterates over batches of examples.
Each forward pass computes predictions from inputs.
The backward pass computes gradients for parameter updates.
Weights are updated using the optimizer.
Training continues until convergence or maximum epochs.
The trained model can be saved and loaded for inference.
Pre-trained models can be fine-tuned for specific tasks.
Transfer learning reduces the need for large training datasets.
Fine-tuning adapts pre-trained knowledge to new domains.
Evaluation metrics assess model performance.
Accuracy measures correct predictions.
Precision and recall evaluate classification quality.
F1 score combines precision and recall.
Perplexity measures language model uncertainty.
BLEU score evaluates machine translation quality.
ROUGE score assesses summarization performance.
The model architecture defines the network structure.
More parameters generally increase model capacity.
Larger models require more computation and data.
Model compression reduces size for deployment.
Quantization reduces precision of weights.
Pruning removes unnecessary connections.
Knowledge distillation trains a smaller student model.
The future of AI holds exciting possibilities.
Artificial general intelligence remains a long-term goal.
Ethical AI ensures fair and beneficial outcomes.
AI safety research prevents harmful applications.
Human-centered design puts people first.
Collaborative AI augments human capabilities.
The journey of AI continues with each breakthrough.
Learning from data is the core principle.
Neural networks are inspired by biological brains.
Deep learning enables hierarchical feature learning.
Representation learning discovers underlying patterns.
Unsupervised learning finds structure without labels.
Supervised learning uses labeled training data.
Semi-supervised learning combines labeled and unlabeled data.
Self-supervised learning creates its own labels.
Contrastive learning learns by comparing examples.
Generative models create new content.
Discriminative models classify existing content.
VAEs learn latent representations.
GANs generate realistic samples.
Diffusion models create high-quality images.
Language models predict next tokens.
Code models understand programming languages.
Multilingual models translate between languages.
Vision models recognize objects and scenes.
Multimodal models understand multiple modalities.
The transformer architecture powers modern AI.
Attention is all you need.
Self-attention computes relationships between positions.
Multi-head attention captures different relationship types.
Positional encoding adds sequence order information.
Layer normalization stabilizes training.
Residual connections enable deeper networks.
The feed-forward network processes attention outputs.
Embedding layers map discrete tokens to vectors.
The softmax function creates probability distributions.
Temperature controls randomness of sampling.
Top-k sampling limits vocabulary consideration.
Nucleus sampling balances quality and diversity.
Beam search explores multiple generation paths.
Greedy decoding always picks most likely token.
Random sampling adds creativity to generation.
The model can be used for various tasks.
Text generation is one common application.
Question answering extracts information from context.
Text summarization condenses long documents.
Sentiment analysis detects emotional tone.
Named entity recognition identifies proper nouns.
Part of speech tagging labels grammatical categories.
Parsing analyzes syntactic structure.
Machine translation converts between languages.
Text classification assigns categories to documents.
The pre-trained model captures general language knowledge.
Fine-tuning adapts it to specific tasks.
Prompt engineering crafts effective inputs.
Few-shot learning learns from few examples.
Zero-shot learning generalizes without training.
In-context learning uses examples in the prompt.
Chain of thought encourages reasoning.
Tool use extends model capabilities.
Retrieval augments generation with external knowledge.
The model size is optimized for efficiency.
Memory usage is important for deployment.
Inference speed affects user experience.
Batch processing improves throughput.
Quantization reduces memory requirements.
Distillation creates smaller models.
Pruning removes redundant weights.
The 16MB constraint shapes the design.
Careful architecture choices enable efficiency.
The model achieves good performance within limits.
Pre-training provides strong initialization.
Fine-tuning adapts to specific needs.
The AI is ready for various applications.
"""


class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self.char_to_idx['<pad>'] = 0
        self.idx_to_char[0] = '<pad>'
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '<pad>') for t in tokens])


def quick_train(model, text, tokenizer, epochs=5):
    """Quick training with small batches"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    tokens = tokenizer.encode(text)
    seq_len = 32
    batch_size = 16
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            
            # Create batch
            x_batch = torch.tensor([x], dtype=torch.long)
            y_batch = torch.tensor([y], dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits.view(-1, model.vocab_size), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return losses


def main():
    print("=" * 60)
    print("Mini AI Pre-training")
    print("=" * 60)
    
    # Get training data
    text = get_training_text()
    print(f"Training text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model - smaller for faster training
    model = MiniAI(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        max_len=128
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / (1024*1024):.2f} MB")
    
    # Quick train
    print("\nStarting quick training...")
    losses = quick_train(model, text, tokenizer, epochs=5)
    
    # Save model
    save_path = "pretrained"
    os.makedirs(save_path, exist_ok=True)
    
    # Save tokenizer
    with open(os.path.join(save_path, 'tokenizer.json'), 'w') as f:
        json.dump({
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': {str(k): v for k, v in tokenizer.idx_to_char.items()}
        }, f)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_heads': 4,
            'num_layers': 4,
            'max_len': 128
        },
        'losses': losses
    }, os.path.join(save_path, 'model.pt'))
    
    # Check final size
    model_size_mb = os.path.getsize(os.path.join(save_path, 'model.pt')) / (1024 * 1024)
    print(f"\nModel saved to {save_path}")
    print(f"  Saved model size: {model_size_mb:.2f} MB")
    print(f"  Total parameters: {total_params:,}")
    
    print("\nPre-training complete!")


if __name__ == '__main__':
    main()
