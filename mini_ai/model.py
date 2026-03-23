"""
Mini AI - A Pre-trained 16MB Neural Network Model
A compact transformer-based language model with pre-trained weights
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import List, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class MiniAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, T, C)
        return self.out(x)

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model: int, d_ff: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MiniAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MiniAI(nn.Module):
    """
    Mini AI - A compact pre-trained language model
    Designed to fit within 16MB with reasonable capabilities
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        max_len: int = 256
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with sensible defaults"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text given input tokens"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if needed
                input_ids_cond = input_ids if input_ids.size(1) <= 256 else input_ids[:, -256:]
                
                logits = self.forward(input_ids_cond)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids
    
    def get_model_size(self) -> int:
        """Get model size in bytes"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    def save_pretrained(self, path: str):
        """Save model and weights"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        
        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': 4,
            'num_layers': 6,
            'max_len': 256
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
            
    @classmethod
    def from_pretrained(cls, path: str) -> 'MiniAI':
        """Load pre-trained model"""
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        return model


def count_parameters(model: nn.Module) -> int:
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size(model: nn.Module) -> float:
    """Estimate model size in MB"""
    size_bytes = 0
    for param in model.parameters():
        size_bytes += param.numel() * param.element_size()
    return size_bytes / (1024 * 1024)


if __name__ == '__main__':
    # Create and test the model
    model = MiniAI()
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {estimate_model_size(model):.2f} MB")
    
    # Test forward pass
    test_input = torch.randint(0, model.vocab_size, (1, 10))
    test_output = model(test_input)
    print(f"Output shape: {test_output.shape}")
