#!/usr/bin/env python3
"""
Quantum GPT Training Script
Trains the quantum-enhanced GPT on Shakespeare dataset with the same interface as original
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from quantum_gpt_implementation import *

def main():
    """Main training function - replicates original GPT training loop with quantum enhancements"""

    print("🚀 Quantum-Enhanced GPT Training")
    print("=" * 50)

    # Load Shakespeare dataset (same as original)
    print("Loading Shakespeare dataset...")
    try:
        with open('data/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: data/input.txt not found!")
        print("Please download the tiny Shakespeare dataset:")
        print("curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt")
        return

    # Create character mappings (same as original)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]  # encoder: string -> list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: list of integers -> string

    # Prepare training data (same as original)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    print(f"Dataset loaded: {len(text)} characters, {vocab_size} unique")
    print(f"Training data: {len(train_data)} tokens")
    print(f"Validation data: {len(val_data)} tokens")

    # Training hyperparameters
    max_iters = 10000  # Reduced for demo
    eval_interval = 100
    eval_iters = 50   # Reduced for quantum simulation
    device = "cpu"
    QISKIT_AVAILABLE = True

    print(f"\nModel Configuration:")
    print(f"  - Embedding dimension: {n_embd}")
    print(f"  - Number of heads: {n_head}")
    print(f"  - Number of layers: {n_layer}")
    print(f"  - Block size: {block_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Quantum qubits: {n_qubits}")
    print(f"  - Qiskit available: {QISKIT_AVAILABLE}")

    # Create model
    model = QuantumGPT(vocab_size)
    model = model.to(device)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params/1e6:.2f}M")

    # Create optimizer (same as original)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\nStarting training for {max_iters} iterations...")
    print("-" * 50)

    # Training loop (same structure as original)
    start_time = time.time()

    for iter in range(max_iters):
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters)
            elapsed = time.time() - start_time
            print(f"step {iter:4d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, time {elapsed:.1f}s")

        # Sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    # Generate from the model (same as original)
    print("\nGenerating text from trained model...")
    print("-" * 50)

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    generated_text = decode(generated_tokens)

    print(generated_text)

    # Save the model
    print("\nSaving trained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'model_config': {
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'n_qubits': n_qubits
        }
    }, 'models/quantum_gpt_model.pth')
    print("Model saved as 'models/quantum_gpt_model.pth'")

    return model, encode, decode

def load_model(model_path='models/quantum_gpt_model.pth'):
    """Load a saved quantum GPT model"""
    checkpoint = torch.load(model_path, map_location="cpu")

    vocab_size = checkpoint['vocab_size']
    model = QuantumGPT(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"Model loaded from {model_path}")
    print(f"Vocabulary size: {vocab_size}")

    return model, encode, decode

def generate_text(model, encode, decode, prompt="", max_new_tokens=200):
    """Generate text using the trained model"""
    model.eval()

    # determine device from model parameters
    device = next(model.parameters()).device

    if prompt:
        # Encode the prompt using provided encode function
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=max_new_tokens)
    generated_text = decode(generated[0].tolist())

    return generated_text

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            # Generate text from saved model
            try:
                model, encode, decode = load_model()
                prompt = sys.argv[2] if len(sys.argv) > 2 else ""
                text = generate_text(model, encode, decode, prompt)
                print("Generated text:")
                print("-" * 50)
                print(text)
            except FileNotFoundError:
                print("No saved model found. Run training first.")
        else:
            print("Usage: python train_quantum_gpt.py [generate] [prompt]")
    else:
        # Run training
        main()
