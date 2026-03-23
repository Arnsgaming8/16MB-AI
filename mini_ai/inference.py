"""
Inference script for Mini AI
Load pre-trained model and generate text
"""

import torch
import torch.nn as nn
import json
import os
from typing import List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MiniAI


class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, char_to_idx: dict, idx_to_char: dict):
        self.char_to_idx = char_to_idx
        self.idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.idx_to_char.get(t, '<pad>') for t in tokens])


def load_model(model_path: str = "pretrained"):
    """Load pre-trained model and tokenizer"""
    # Load tokenizer
    with open(os.path.join(model_path, 'tokenizer.json'), 'r') as f:
        tokenizer_data = json.load(f)
    
    tokenizer = SimpleTokenizer(
        tokenizer_data['char_to_idx'],
        tokenizer_data['idx_to_char']
    )
    
    # Load model
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
    config = checkpoint['config']
    
    model = MiniAI(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_len=config['max_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def generate_text(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40
) -> str:
    """Generate text from prompt"""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode output
    output_tokens = output_ids[0].tolist()
    return tokenizer.decode(output_tokens)


def interactive_mode(model: nn.Module, tokenizer: SimpleTokenizer):
    """Run interactive mode"""
    print("\n" + "=" * 60)
    print("Mini AI Interactive Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nMini AI: ", end="")
            response = generate_text(model, tokenizer, prompt, max_new_tokens=150)
            # Only show the newly generated part
            response = response[len(prompt):]
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_mode(model: nn.Module, tokenizer: SimpleTokenizer):
    """Run demo mode with predefined prompts"""
    prompts = [
        "The quick brown",
        "Artificial intelligence",
        "Machine learning",
        "Deep neural networks",
        "Natural language processing",
    ]
    
    print("\n" + "=" * 60)
    print("Mini AI Demo Mode")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 40)
        response = generate_text(model, tokenizer, prompt, max_new_tokens=80)
        print(response[len(prompt):])
        print()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mini AI Inference')
    parser.add_argument('--model_path', type=str, default='pretrained',
                        help='Path to pre-trained model')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling parameter')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please run train.py first to create pre-trained weights")
        return
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded successfully!")
    
    # Get model size
    param_size = sum(p.numel() for p in model.parameters()) * 4  # float32
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {param_size / (1024*1024):.2f} MB")
    
    # Run based on mode
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.demo:
        demo_mode(model, tokenizer)
    elif args.prompt:
        print(f"\nPrompt: \"{args.prompt}\"")
        print("-" * 40)
        response = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(response[len(args.prompt):])
    else:
        # Default to demo
        demo_mode(model, tokenizer)


if __name__ == '__main__':
    main()
