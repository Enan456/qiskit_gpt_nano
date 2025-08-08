
#!/usr/bin/env python3
"""
Quantum-Enhanced GPT Test Script
Demonstrates integration of Qiskit quantum computing with transformer architecture
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from quantum_gpt_implementation import *

# Set random seeds for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

def create_dummy_dataset(vocab_size=65, seq_length=1000):
    """Create a simple dummy dataset for testing"""
    # Create a simple pattern-based dataset
    data = []
    patterns = [
        [1, 2, 3, 4],  # Simple sequence
        [5, 6, 7, 8, 9],  # Another sequence
        [10, 11, 12],  # Short sequence
    ]

    for _ in range(seq_length):
        pattern = patterns[np.random.randint(0, len(patterns))]
        data.extend(pattern)

    # Add some random tokens
    for _ in range(100):
        data.append(np.random.randint(0, vocab_size))

    return torch.tensor(data[:seq_length], dtype=torch.long)

def test_quantum_attention():
    """Test the quantum attention head"""
    print("\n=== Testing Quantum Attention Head ===")

    try:
        # Create a quantum attention head
        head = QuantumAttentionHead(head_size=8, n_qubits=4)

        # Test input
        batch_size, seq_len, embed_dim = 2, 6, 16
        x = torch.randn(batch_size, seq_len, embed_dim)

        print(f"Input shape: {x.shape}")

        # Forward pass
        output = head(x)
        print(f"Quantum attention output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Check if quantum enhancement is working
        if hasattr(head, 'quantum_scale'):
            print(f"Quantum scaling parameter: {head.quantum_scale.item():.4f}")

        return True

    except Exception as e:
        print(f"Quantum attention test failed: {e}")
        return False

def test_quantum_model():
    """Test the complete quantum GPT model"""
    print("\n=== Testing Quantum GPT Model ===")

    try:
        vocab_size = 65
        model = QuantumGPT(vocab_size)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Test forward pass
        batch_size, seq_len = 2, 8
        test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        test_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        print(f"Test input shape: {test_input.shape}")

        logits, loss = model(test_input, test_targets)
        print(f"Output logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")

        # Test generation
        print("\nTesting text generation...")
        context = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(context, max_new_tokens=10)
        print(f"Generated sequence: {generated[0].tolist()}")

        return True

    except Exception as e:
        print(f"Quantum model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_classical_vs_quantum():
    """Compare classical vs quantum attention performance"""
    print("\n=== Comparing Classical vs Quantum Attention ===")

    try:
        # Create heads
        classical_head = ClassicalHead(head_size=8)
        quantum_head = QuantumAttentionHead(head_size=8, n_qubits=4)

        # Test input
        x = torch.randn(2, 6, 16)

        # Classical attention
        classical_output = classical_head(x)
        classical_variance = torch.var(classical_output).item()

        # Quantum attention  
        quantum_output = quantum_head(x)
        quantum_variance = torch.var(quantum_output).item()

        print(f"Classical attention variance: {classical_variance:.6f}")
        print(f"Quantum attention variance: {quantum_variance:.6f}")

        # Check if outputs are different
        diff = torch.mean(torch.abs(classical_output - quantum_output)).item()
        print(f"Mean absolute difference: {diff:.6f}")

        if diff > 1e-6:
            print("✓ Quantum enhancement is modifying attention patterns")
        else:
            print("⚠ Quantum enhancement may not be significant")

        return True

    except Exception as e:
        print(f"Comparison test failed: {e}")
        return False

def mini_training_demo():
    """Demonstrate training the quantum GPT on a small dataset"""
    print("\n=== Mini Training Demonstration ===")

    try:
        # Create model and data
        vocab_size = 20  # Small vocab for demo
        model = QuantumGPT(vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create simple training data
        data = create_dummy_dataset(vocab_size, seq_length=200)

        print(f"Training data length: {len(data)}")
        print(f"Sample data: {data[:20].tolist()}")

        # Training loop
        model.train()
        losses = []

        for step in range(20):  # Very short training
            # Get batch
            X, Y = get_batch(data, batch_size=2, block_size=8)

            # Forward pass
            logits, loss = model(X, Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 5 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")

        # Test generation after training
        context = torch.tensor([[1]], dtype=torch.long)  
        generated = model.generate(context, max_new_tokens=15)
        print(f"Generated after training: {generated[0].tolist()}")

        return True

    except Exception as e:
        print(f"Training demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Quantum-Enhanced GPT Testing Suite")
    print("=" * 50)

    # Check if Qiskit is available
    try:
        import qiskit
        print(f"Qiskit version: {qiskit.__version__}")
        import qiskit_machine_learning
        print("Qiskit Machine Learning available")
    except ImportError as e:
        print(f"Warning: Qiskit not available: {e}")
        print("Install with: pip install qiskit qiskit-machine-learning")
        return

    # Run tests
    tests = [
        ("Quantum Attention Head", test_quantum_attention),
        ("Quantum GPT Model", test_quantum_model), 
        ("Classical vs Quantum Comparison", compare_classical_vs_quantum),
        ("Mini Training Demo", mini_training_demo),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test '{test_name}' failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<35} {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your quantum GPT is ready for experimentation.")
    else:
        print("⚠️  Some tests failed. Check error messages above.")

if __name__ == "__main__":
    main()
