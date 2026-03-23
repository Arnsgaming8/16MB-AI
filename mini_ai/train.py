"""
Training script for Mini AI
Pre-trains the model and saves weights
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List
import json

from model import MiniAI, count_parameters, estimate_model_size


class TextDataset(Dataset):
    """Simple text dataset for pre-training"""
    def __init__(self, text: str, tokenizer, seq_len: int = 64):
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self.char_to_idx['<pad>'] = 0
        self.idx_to_char[0] = '<pad>'
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.idx_to_char.get(t, '<pad>') for t in tokens])


def get_training_text() -> str:
    """Get training text data - a mix of common patterns"""
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu',
    epochs: int = 10
) -> List[float]:
    """Train the model"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return losses


def main():
    """Main training function"""
    print("=" * 60)
    print("Mini AI Pre-training")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get training data
    text = get_training_text()
    print(f"Training text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = MiniAI(
        vocab_size=tokenizer.vocab_size,
        d_model=192,
        num_heads=4,
        num_layers=6,
        max_len=128
    )
    model = model.to(device)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Model size: {estimate_model_size(model):.2f} MB")
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, seq_len=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"  Training batches: {len(train_loader)}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\nStarting training...")
    losses = train_model(model, train_loader, optimizer, criterion, device, epochs=10)
    
    # Save model
    save_path = "pretrained"
    print(f"\nSaving model to {save_path}...")
    
    # Save tokenizer
    with open(os.path.join(save_path, 'tokenizer.json'), 'w') as f:
        json.dump({
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': {str(k): v for k, v in tokenizer.idx_to_char.items()}
        }, f)
    
    # Save model
    model = model.cpu()
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_heads': 4,
            'num_layers': 6,
            'max_len': 128
        },
        'losses': losses
    }, os.path.join(save_path, 'model.pt'))
    
    # Check final size
    model_size_mb = os.path.getsize(os.path.join(save_path, 'model.pt')) / (1024 * 1024)
    print(f"  Saved model size: {model_size_mb:.2f} MB")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Final model size: {estimate_model_size(model):.2f} MB")
    
    print("\nPre-training complete!")
    return model, tokenizer


if __name__ == '__main__':
    main()
